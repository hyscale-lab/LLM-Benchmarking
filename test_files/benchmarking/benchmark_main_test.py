import pytest
import matplotlib

matplotlib.use("Agg")
import os
from unittest.mock import patch, MagicMock
from benchmarking.benchmark_main import Benchmark
from datetime import datetime


class MockProvider:
    """Mock provider class to simulate provider behavior."""

    def __init__(self, name, model_map):
        self.name = name
        self.metrics = {
            "response_times": {model: [] for model in model_map},
            "timetofirsttoken": {model: [] for model in model_map},
            "timebetweentokens": {model: [] for model in model_map},
            "timebetweentokens_median": {model: [] for model in model_map},
            "timebetweentokens_p95": {model: [] for model in model_map},
        }
        self.model_map = model_map

    def initialize_client(self):
        pass

    def log_metrics(self, model_name, metric, value):
        pass

    def get_model_name(self, model):
        return self.model_map.get(model, None)

    def perform_inference(self, model, prompt, max_output, verbosity):
        pass

    def perform_inference_streaming(self, model, prompt, max_output, verbosity):
        pass


@pytest.fixture
def setup_benchmark():
    """Fixture to initialize the Benchmark class with mock providers."""
    providers = [
        MockProvider("Provider1", {"model_a": "Model A", "model_b": "Model B"}),
        MockProvider("Provider2", {"model_a": "Model A", "model_b": "Model B"}),
    ]
    num_requests = 2
    models = ["model_a", "model_b"]
    prompt = "Test prompt"
    max_output = 100
    verbosity = True
    return Benchmark(
        providers,
        num_requests,
        models,
        max_output,
        prompt,
        streaming=False,
        verbosity=verbosity,
    )


def test_benchmark_initialization(setup_benchmark):
    """Test the initialization of Benchmark, including directory creation."""
    benchmark = setup_benchmark
    base_dir = "end_to_end"
    provider_dir_name = "mockprovider_mockprovider"
    expected_graph_dir = os.path.join("benchmark_graph", base_dir, provider_dir_name)

    # Verify graph directory is created correctly
    assert benchmark.graph_dir == expected_graph_dir
    assert os.path.exists(benchmark.graph_dir)


@patch.object(MockProvider, "perform_inference_streaming", return_value=None)
@patch.object(MockProvider, "perform_inference", return_value=None)
def test_benchmark_run_non_streaming(
    mock_perform_inference, mock_perform_inference_streaming, setup_benchmark
):
    """Test the Benchmark run method in non-streaming mode."""
    benchmark = setup_benchmark
    benchmark.run()

    # Ensure perform_inference was called for each provider, model, and request
    total_calls = (
        len(benchmark.providers) * len(benchmark.models) * benchmark.num_requests
    )
    assert mock_perform_inference.call_count == total_calls
    assert mock_perform_inference_streaming.call_count == 0  # Should not be called


@patch.object(MockProvider, "perform_inference_streaming", return_value=None)
@patch.object(MockProvider, "perform_inference", return_value=None)
def test_benchmark_run_streaming(
    mock_perform_inference, mock_perform_inference_streaming
):
    """Test the Benchmark run method in streaming mode."""
    providers = [
        MockProvider("Provider1", {"model_a": "Model A"}),
        MockProvider("Provider2", {"model_a": "Model A"}),
    ]
    benchmark = Benchmark(
        providers, 2, ["model_a"], 100, "Test prompt", streaming=True, verbosity=True
    )
    benchmark.run()

    # Ensure perform_inference_streaming was called for each provider, model, and request
    total_calls = (
        len(benchmark.providers) * len(benchmark.models) * benchmark.num_requests
    )
    assert mock_perform_inference_streaming.call_count == total_calls
    assert (
        mock_perform_inference.call_count == 0
    )  # Should not be called in streaming mode


@patch("benchmarking.benchmark_main.datetime")
@patch("benchmarking.benchmark_main.plt")
def test_plot_metrics(mock_plt, mock_datetime, setup_benchmark):
    """Test the plot_metrics method, ensuring plots are saved with the correct filename."""
    benchmark = setup_benchmark
    mock_datetime.now.return_value = datetime(2023, 11, 1, 12, 0, 0)
    benchmark.providers[0].metrics["response_times"]["model_a"] = [0.1, 0.2, 0.3]
    benchmark.providers[0].metrics["response_times"]["model_b"] = [0.15, 0.25, 0.35]

    benchmark.plot_metrics("response_times", "response_times")

    # Verify the filename and path are generated correctly
    expected_filename = os.path.join(
        benchmark.graph_dir, "response_times_231101_1200.png"
    )
    mock_plt.savefig.assert_called_once_with(expected_filename)
    mock_plt.close.assert_called_once()

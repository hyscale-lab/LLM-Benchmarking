import pytest
from unittest.mock import MagicMock,patch
import os
import json
import numpy as np
from benchmarking.dynamo_bench import Benchmark

# Mock provider for testing
class MockProvider:
    def __init__(self):
        self.metrics = {
            "response_times": {"mock_model": [0.123, 0.456, 0.789]},
            "timetofirsttoken": {"mock_model": [0.05, 0.07, 0.09]},
            "timebetweentokens": {"mock_model": [0.02, 0.03, 0.04]},
            "timebetweentokens_median": {"mock_model": [0.02]},
            "timebetweentokens_p95": {"mock_model": [0.04]},
        }

    def log_metrics(self, model_name, metric, value):
        pass
        
    def get_model_name(self, model):
        return model

    def perform_inference(self, model, prompt, max_output, verbosity):
        pass

    def perform_inference_streaming(self, model, prompt, max_output, verbosity):
        pass

@pytest.fixture
def benchmark_instance():
    os.environ["AWS_REGION"] = "us-east-1"
    os.environ["DYNAMODB_ENDPOINT_URL"] = "http://localhost:8000"
    
    providers = [MockProvider()]
    num_requests = 1
    models = ["mock_model"]
    prompt = "test prompt"
    max_output = 100
    verbosity = True
    graph_dir='/'

    # Mock DynamoDB
    mock_dynamodb = MagicMock()
    Benchmark.dynamodb = mock_dynamodb

    return Benchmark(
        providers,
        num_requests,
        models,
        max_output,
        prompt,
        streaming=False,
        verbosity=verbosity,
    )

def test_initialization(benchmark_instance):
    assert benchmark_instance.prompt == "test prompt"
    assert benchmark_instance.num_requests == 1
    assert benchmark_instance.max_output == 100
    assert benchmark_instance.verbosity is True
    assert isinstance(benchmark_instance.run_id, str)
    assert benchmark_instance.benchmark_data["prompt"] == "test prompt"
    assert "providers" in benchmark_instance.benchmark_data

def test_add_metric_data(benchmark_instance):
    provider_name = "MockProvider"
    model_name = "mock_model"
    metric = "response_times"
    latencies = [0.1, 0.2, 0.3]
    
    benchmark_instance.add_metric_data(provider_name, model_name, metric, latencies)
    
    data = benchmark_instance.benchmark_data["providers"][provider_name][model_name][metric]
    assert "latencies" in data
    assert "cdf" in data
    assert all(isinstance(val, str) for val in data["latencies"])
    assert all(isinstance(val, str) for val in data["cdf"])

def test_clean_data(benchmark_instance):
    data = {
        "valid": "value",
        "empty_string": "",
        "none_value": None,
        "empty_list": [],
        "nested": {"empty_dict": {}, "valid": "nested_value"},
        "float_value": 0.1234,
    }
    cleaned_data = benchmark_instance.clean_data(data)
    
    assert "valid" in cleaned_data
    assert "nested" in cleaned_data and "valid" in cleaned_data["nested"]
    assert "empty_string" not in cleaned_data
    assert "none_value" not in cleaned_data
    assert "empty_list" not in cleaned_data
    assert "empty_dict" not in cleaned_data["nested"]
    assert isinstance(cleaned_data["float_value"], str)

def test_store_data_points(benchmark_instance):
    mock_table = MagicMock()
    benchmark_instance.dynamodb.Table = MagicMock(return_value=mock_table)
    mock_table.put_item = MagicMock()
    
    provider_name = "MockProvider"
    model_name = "mock_model"
    metric = "response_times"
    latencies = [0.1, 0.2, 0.3]
    
    benchmark_instance.add_metric_data(provider_name, model_name, metric, latencies)
    benchmark_instance.store_data_points()
    
    mock_table.put_item.assert_called_once()
    args, kwargs = mock_table.put_item.call_args
    item = kwargs["Item"]
    
    assert item["run_id"] == benchmark_instance.run_id
    assert item["provider_name"] == provider_name
    assert item["model_name"] == model_name
    assert item["prompt"] == benchmark_instance.prompt
    assert isinstance(item["metrics"], str)

def test_plot_metrics(benchmark_instance):
    # Mock provider setup
    mock_provider = MockProvider()
    provider_name = mock_provider.__class__.__name__
    model_name = "mock_model"
    
    # Manually add mock data to benchmark_data
    benchmark_instance.benchmark_data["providers"] = {
        provider_name: {
            model_name: {
                "response_times": {
                    "latencies": [100, 200, 300],
                    "cdf": [0.33, 0.66, 1.0]
                }
            }
        }
    }
    
    # Mock the savefig method
    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        benchmark_instance.graph_dir = "/tmp"  # Ensure this directory exists
        benchmark_instance.plot_metrics("response_times")
        mock_savefig.assert_called_once()  # Ensure savefig was called once

    # Verify that benchmark_data has been populated correctly
    assert provider_name in benchmark_instance.benchmark_data["providers"]
    assert model_name in benchmark_instance.benchmark_data["providers"][provider_name]
    assert "response_times" in benchmark_instance.benchmark_data["providers"][provider_name][model_name]

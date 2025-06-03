import unittest
import json
from unittest.mock import patch, MagicMock
import os
import tempfile
import sys
import main
from main import parser

from main import (
    load_config,
    validate_providers,
    get_common_models,
    validate_selected_models,
    input_sizes,
    OUTPUT_SIZE_LOWER_LIMIT,
    OUTPUT_SIZE_UPPER_LIMIT,
    run_benchmark,
)


class TestMain(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""

        self.env_patcher = patch.dict(
            os.environ,
            {
                "CLOUDFLARE_ACCOUNT_ID": "test_account_id",
                "CLOUDFLARE_AI_TOKEN": "test_api_token",
                "TOGETHER_AI_API": "test_together_api",
                "OPEN_AI_API": "test_openai_api",
                "ANTHROPIC_API": "test_anthropic_api_key",
                "PERPLEXITY_AI_API": "test_api_perplexity",
                "GROQ_API_KEY": "test_groq_key",
                "GEMINI_API_KEY": "test_gemini_key",
                "HYPERBOLIC_API": "hyperbolic_test_api",
                "AWS_BEDROCK_ACCESS_KEY_ID": "aws_test_id",
                "AWS_BEDROCK_SECRET_ACCESS_KEY": "aws_test_key",
                "AWS_BEDROCK_REGION": "us-east-1",
                "AZURE_API_KEY": "test_azure_api",
            },
        )

        self.env_patcher.start()

        self.sample_config = {
            "providers": ["TogetherAI", "OpenAI"],
            "num_requests": 2,
            "models": ["meta-llama-3.2-3b-instruct"],
            "input_tokens": 100,
            "streaming": False,
            "max_output": 1000,
            "verbose": True,
        }

        # Create a temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(self.config_path, "w") as f:
            json.dump(self.sample_config, f)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.temp_dir.cleanup()

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config = load_config(self.config_path)
        self.assertIsInstance(config, dict)
        self.assertIn("providers", config)
        self.assertIn("num_requests", config)
        self.assertIn("models", config)
        self.assertEqual(config["providers"], ["TogetherAI", "OpenAI"])

    def test_load_nonexistent_config(self):
        """Test loading a nonexistent configuration file."""
        config = load_config("nonexistent.json")
        self.assertIsNone(config)

    def test_load_invalid_json(self):
        """Test loading an invalid JSON file."""
        invalid_path = os.path.join(self.temp_dir.name, "invalid.json")
        with open(invalid_path, "w") as f:
            f.write("invalid json content")
        config = load_config(invalid_path)
        self.assertIsNone(config)

    def test_validate_valid_providers(self):
        """Test validation of valid providers."""
        providers = ["TogetherAI", "OpenAI"]
        valid_providers = validate_providers(providers)
        self.assertEqual(len(valid_providers), 2)
        provider_names = [provider.__class__.__name__ for provider in valid_providers]
        print(provider_names)
        self.assertIn("TogetherAI", provider_names)
        self.assertIn("Open_AI", provider_names)

    def test_validate_invalid_providers(self):
        """Test validation with invalid providers."""
        providers = ["InvalidProvider", "TogetherAI"]
        valid_providers = validate_providers(providers)
        self.assertEqual(len(valid_providers), 1)
        self.assertEqual(valid_providers[0].__class__.__name__, "TogetherAI")

    def test_validate_empty_providers(self):
        """Test validation with empty provider list."""
        providers = []
        valid_providers = validate_providers(providers)
        self.assertEqual(len(valid_providers), 0)

    def test_get_common_models(self):
        """Test getting common models across providers."""
        # Create mock providers
        provider1 = MagicMock()
        provider1.model_map = {
            "model1": "actual_model1",
            "model2": "actual_model2",
            "common_model": "actual_common",
        }

        provider2 = MagicMock()
        provider2.model_map = {
            "model3": "actual_model3",
            "model4": "actual_model4",
            "common_model": "actual_common",
        }

        mock_providers = [provider1, provider2]
        common_models = get_common_models(mock_providers)
        self.assertIn("common_model", common_models)
        self.assertEqual(len(common_models), 1)

    def test_get_common_models_no_providers(self):
        """Test getting common models with no providers."""
        common_models = get_common_models([])
        self.assertEqual(len(common_models), 0)

    def test_validate_selected_models_common(self):
        """Test validation of selected models with common models."""
        provider1 = MagicMock()
        provider1.model_map = {
            "common_model": "actual_common",
            "model1": "actual_model1",
        }
        provider2 = MagicMock()
        provider2.model_map = {
            "common_model": "actual_common",
            "model2": "actual_model2",
        }

        mock_providers = [provider1, provider2]
        selected_models = ["common_model", "invalid_model"]
        common_models = get_common_models(mock_providers)
        valid_models = validate_selected_models(
            selected_models, common_models, mock_providers
        )

        self.assertEqual(len(valid_models), 1)
        self.assertIn("common_model", valid_models)

    def test_validate_selected_models_single_provider(self):
        """Test validation of selected models with a single provider."""
        provider = MagicMock()
        provider.model_map = {"model1": "actual_model1", "model2": "actual_model2"}

        selected_models = ["model1", "model2"]
        valid_models = validate_selected_models(selected_models, [], [provider])

        self.assertEqual(len(valid_models), 2)
        self.assertIn("model1", valid_models)
        self.assertIn("model2", valid_models)

    def test_validate_selected_models_no_valid_models(self):
        """Test validation where no models are valid in common or provider model_map."""
        provider = MagicMock()
        provider.model_map = {"model1": "actual_model1", "model2": "actual_model2"}

        selected_models = ["invalid_model"]
        common_models = []
        valid_models = validate_selected_models(
            selected_models, common_models, [provider]
        )

        self.assertEqual(len(valid_models), 0)  # Ensure valid_models is empty

    def test_validate_selected_models_no_common_models(self):
        """Test validation when models are not common to all providers."""
        provider1 = MagicMock()
        provider1.model_map = {
            "model1": "actual_model1",
            "common_model": "actual_common",
        }
        provider2 = MagicMock()
        provider2.model_map = {"model2": "actual_model2"}

        selected_models = ["model1"]
        common_models = get_common_models(
            [provider1, provider2]
        )  # This should be empty
        valid_models = validate_selected_models(
            selected_models, common_models, [provider1, provider2]
        )

        self.assertEqual(valid_models, [])  # Ensure valid_models is empty

    def test_input_size_validation(self):
        """Test input size validation."""
        self.assertIn(self.sample_config["input_tokens"], input_sizes)

        invalid_config = self.sample_config.copy()
        invalid_config["input_tokens"] = 42  # Invalid input size
        with patch("benchmarking.benchmark_main.Benchmark") as mock_benchmark:
            run_benchmark(invalid_config)
            mock_benchmark.assert_not_called()

    def test_output_size_validation(self):
        """Test output size validation."""
        self.assertGreaterEqual(
            self.sample_config["max_output"], OUTPUT_SIZE_LOWER_LIMIT
        )
        self.assertLessEqual(self.sample_config["max_output"], OUTPUT_SIZE_UPPER_LIMIT)

        invalid_config = self.sample_config.copy()
        invalid_config["max_output"] = (
            OUTPUT_SIZE_UPPER_LIMIT + 1
        )  # Invalid output size
        with patch("benchmarking.benchmark_main.Benchmark") as mock_benchmark:
            run_benchmark(invalid_config)
            mock_benchmark.assert_not_called()

    @patch("benchmarking.benchmark_main.Benchmark")
    @patch("main.get_prompt")
    def test_run_benchmark_success(self, mock_get_prompt, mock_benchmark):
        """Test successful benchmark execution."""
        mock_get_prompt.return_value = "test prompt"
        mock_benchmark_instance = MagicMock()
        mock_benchmark.return_value = mock_benchmark_instance

        run_benchmark(self.sample_config)

        mock_benchmark.assert_called_once()
        mock_benchmark_instance.run.assert_called_once()

    @patch("benchmarking.benchmark_main.Benchmark")
    def test_run_benchmark_invalid_config(self, mock_benchmark):
        """Test benchmark execution with invalid configuration."""
        invalid_config = self.sample_config.copy()
        invalid_config["providers"] = ["InvalidProvider"]

        run_benchmark(invalid_config)
        mock_benchmark.assert_not_called()


class TestCommandLineInterface(unittest.TestCase):
    @patch("main.display_available_providers")
    def test_list_flag(self, mock_display_providers):
        """Test --list flag functionality"""
        test_args = ["main.py", "--list"]
        with patch.object(sys, "argv", test_args):
            main.__name__ = "__main__"
            main.main()  # Assuming there is a main() function in main.py

        # Verify display_available_providers was called
        mock_display_providers.assert_called_once()

    @patch("main.run_benchmark")
    @patch("main.load_config", return_value={"providers": ["TogetherAI"]})
    def test_config_flag(self, mock_load_config, mock_run_benchmark):
        """Test -c/--config flag functionality"""
        test_args = ["main.py", "-c", "config.json", "--vllm_ip", "192.168.1.10"]
        with patch.object(sys, "argv", test_args):
            main.__name__ = "__main__"
            main.main()

        # Verify the config was loaded and benchmark was run
        mock_load_config.assert_called_once_with("config.json")
        mock_run_benchmark.assert_called_once_with(mock_load_config.return_value, '192.168.1.10')

    @patch("main.run_benchmark")
    @patch("main.load_config", return_value=None)
    def test_config_load_failure(self, mock_load_config, mock_run_benchmark):
        """Test behavior when config loading fails"""
        test_args = ["main.py", "-c", "config.json"]
        with patch.object(sys, "argv", test_args):
            main.__name__ = "__main__"
            main.main()

        # Verify benchmark was not run
        mock_run_benchmark.assert_not_called()

    @patch("argparse.ArgumentParser.print_help")
    def test_no_args(self, mock_print_help):
        """Test behavior when no arguments are provided"""
        test_args = ["main.py"]
        with patch.object(sys, "argv", test_args):
            main.__name__ = "__main__"
            main.main()

        # Verify help was printed
        mock_print_help.assert_called_once()

    def test_argument_parser_creation(self):
        """Test the creation of argument parser with correct arguments"""
        # Test parser description
        self.assertEqual(
            parser.description, "Run a benchmark on selected AI providers and models."
        )

        # Test that the parser has the expected arguments
        args = vars(parser.parse_args([]))
        self.assertIn("config", args)
        self.assertIn("list", args)

        # Test argument types
        self.assertIsNone(args["config"])
        self.assertFalse(args["list"])

        # Test --list flag
        args = vars(parser.parse_args(["--list"]))
        self.assertTrue(args["list"])

        # Test --config flag
        args = vars(parser.parse_args(["-c", "test.json"]))
        self.assertEqual(args["config"], "test.json")


if __name__ == "__main__":
    unittest.main()

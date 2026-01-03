import unittest
from unittest.mock import patch, MagicMock
from providers import PerplexityAI


class TestPerplexityAI(unittest.TestCase):

    @patch("os.environ.get")
    def test_initialization_with_valid_api_key(self, mock_env_get):
        mock_env_get.return_value = "test_api_key"

        # Mock OpenAI class to avoid making actual API calls
        with patch("providers.perplexity_ai_provider.OpenAI") as mock_openai_client:
            mock_openai_client.return_value = MagicMock()

            # Instantiate the PerplexityAI class
            provider = PerplexityAI()

            # Assert that the client is correctly initialized
            mock_openai_client.assert_called_with(
                api_key="test_api_key", base_url="https://api.perplexity.ai", max_retries=0
            )
            self.assertEqual(provider.client, mock_openai_client.return_value)

            # Assert model map is initialized
            expected_model_map = {
            "sonar": "sonar",  
            "sonar-pro": "sonar-pro", 
            "sonar-reasoning-pro": "sonar-reasoning-pro",  
            "common-model": "sonar-pro",
        }
            self.assertEqual(provider.model_map, expected_model_map)

    @patch("os.environ.get")
    def test_initialization_without_api_key(self, mock_env_get):
        mock_env_get.return_value = None

        with self.assertRaises(ValueError) as context:
            PerplexityAI()

        self.assertEqual(
            str(context.exception),
            "Perplexity AI API token must be provided as an environment variable.",
        )

    @patch("os.environ.get")
    def test_get_model_name(self, mock_env_get):
        # Mock the environment variable for API key
        mock_env_get.return_value = "test_api_key"

        # Mock OpenAI class to avoid making actual API calls
        with patch("providers.perplexity_ai_provider.OpenAI") as mock_openai_client:
            mock_openai_client.return_value = MagicMock()

            # Instantiate the PerplexityAI class
            provider = PerplexityAI()

            self.assertEqual(
                provider.get_model_name("sonar"),
                 "sonar",
            )
            self.assertEqual(
                provider.get_model_name("sonar-pro"),
                 "sonar-pro",
            )
            self.assertEqual(
                provider.get_model_name("sonar-reasoning-pro"),
                "sonar-reasoning-pro",
            )
            self.assertIsNone(provider.get_model_name("non-existent-model"))


if __name__ == "__main__":
    unittest.main()

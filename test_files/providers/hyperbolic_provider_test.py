import pytest
import os
from unittest.mock import patch
from providers.hyperbolic_provider import Hyperbolic


@pytest.fixture
def setup_hyperbolic_provider():
    """Fixture to set up and return an instance of Hyperbolic with a mocked API key."""
    with patch.dict(os.environ, {"HYPERBOLIC_API": "test_api_key"}):
        return Hyperbolic()


def test_hyperbolic_provider_initialization(setup_hyperbolic_provider):
    """Test that Hyperbolic initializes correctly and sets the API key and model_map."""
    provider = setup_hyperbolic_provider

    # Check model_map is set correctly
    assert provider.model_map == {
        "meta-llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
        "meta-llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "common-model": "Qwen/Qwen2.5-VL-7B-Instruct",
        'reasoning-model': [
            'openai/gpt-oss-120b',
        ],
    }


def test_hyperbolic_provider_api_key():
    """Test that Hyperbolic raises an error if HYPERBOLIC_API is missing."""
    with patch.dict(os.environ, {}, clear=True):  # Clear environment variables
        with pytest.raises(
            ValueError,
            match="HYPERBOLIC_API token must be provided as an environment variable.",
        ):
            Hyperbolic()


@patch("providers.hyperbolic_provider.OpenAI")
def test_hyperbolic_provider_client_initialization(mock_openai_client):
    """Test that Hyperbolic initializes the OpenAI client with the correct API key and base URL."""
    with patch.dict(os.environ, {"HYPERBOLIC_API": "test_api_key"}):
        provider = Hyperbolic()

    # Verify the client is created with the correct API key and base URL
    mock_openai_client.assert_called_once_with(
        api_key="test_api_key", base_url="https://api.hyperbolic.xyz/v1"
    )


@patch("providers.hyperbolic_provider.OpenAI")
def test_hyperbolic_provider_get_model_name(
    mock_openai_client, setup_hyperbolic_provider
):
    """Test that get_model_name correctly retrieves model names from model_map."""
    provider = setup_hyperbolic_provider

    # Check model names retrieval
    assert (
        provider.get_model_name("meta-llama-3.2-3b-instruct")
        == "meta-llama/Llama-3.2-3B-Instruct"
    )
    assert (
        provider.get_model_name("qwen2.5-vl-7b-instruct") == "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    assert (
        provider.get_model_name("meta-llama-3.1-70b-instruct")
        == "meta-llama/Meta-Llama-3.1-70B-Instruct"
    )
    assert (
        provider.get_model_name("non-existent-model") is None
    )  # Should return None for unknown model

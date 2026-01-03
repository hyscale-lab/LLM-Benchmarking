import pytest
import os
from unittest.mock import patch
from providers.groq_provider import GroqProvider


@pytest.fixture
def setup_groq_provider():
    """Fixture to set up and return an instance of GroqProvider with a mocked API key."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"}):
        return GroqProvider()


def test_groq_provider_initialization(setup_groq_provider):
    """Test that GroqProvider initializes correctly and sets the API key and model_map."""
    provider = setup_groq_provider

    # Ensure model_map is set correctly
    assert provider.model_map == {
        "google-gemma-7b-it": "gemma-7b-it",
        "meta-llama-3.2-3b-instruct": "llama-3.2-3b-preview",
        "meta-llama-3.1-70b-instruct": "llama-3.3-70b-versatile",
        "common-model": "llama-3.3-70b-versatile",
        'reasoning-model': [
            'openai/gpt-oss-120b',
        ],
    }


def test_groq_provider_api_key():
    """Test that GroqProvider raises an error if GROQ_API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):  # Clear environment variables
        with pytest.raises(KeyError, match="GROQ_API_KEY"):
            GroqProvider()


@patch("providers.groq_provider.Groq")
def test_groq_provider_client_initialization(mock_groq_client):
    """Test that GroqProvider initializes the Groq client with the correct API key."""
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_api_key"}):
        provider = GroqProvider()

    # Ensure the client is created with the correct API key
    mock_groq_client.assert_called_once_with(api_key="test_api_key", max_retries=0)


@patch("providers.groq_provider.Groq")
def test_groq_provider_get_model_name(mock_groq_client, setup_groq_provider):
    """Test that get_model_name correctly retrieves model names from model_map."""
    provider = setup_groq_provider

    # Check model names retrieval
    assert provider.get_model_name("google-gemma-7b-it") == "gemma-7b-it"
    assert (
        provider.get_model_name("meta-llama-3.2-3b-instruct") == "llama-3.2-3b-preview"
    )
    assert (
        provider.get_model_name("meta-llama-3.1-70b-instruct")
        == "llama-3.3-70b-versatile"
    )
    assert (
        provider.get_model_name("non-existent-model") is None
    )  # Should return None for unknown model

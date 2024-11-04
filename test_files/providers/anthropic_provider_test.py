import pytest
import os
from unittest.mock import patch, MagicMock
from providers.anthropic_provider import Anthropic

@pytest.fixture
def setup_anthropic_provider():
    """Fixture to set up and return an instance of Anthropic with a mocked API key."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_api_key"}):
        return Anthropic()

def test_anthropic_provider_initialization(setup_anthropic_provider):
    """Test that Anthropic initializes correctly and sets the API key and model_map."""
    provider = setup_anthropic_provider

    # Ensure model_map is set correctly
    assert provider.model_map == {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-opus": "claude-3-opus-latest",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }

def test_anthropic_provider_api_key():
    """Test that Anthropic raises an error if ANTHROPIC_API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):  # Clear environment variables
        with pytest.raises(ValueError, match="API key must be provided as an environment variable."):
            Anthropic()

@patch("providers.anthropic_provider.anthropic.Anthropic")
def test_perform_inference(mock_anthropic_client_class, setup_anthropic_provider):
    """Test perform_inference calls the correct methods with the correct parameters."""
    provider = setup_anthropic_provider

    # Mock the client instance and replace it in the provider instance
    mock_client_instance = MagicMock()
    mock_response = {"content": [{"text": "Test response"}]}
    mock_client_instance.messages.create.return_value = mock_response
    provider.client = mock_client_instance  # Directly set the mock client

    # Call the method with verbosity enabled
    elapsed_time = provider.perform_inference("claude-3.5-sonnet", "Test prompt", max_output=100, verbosity=True)

    # Verify messages.create is called with correct parameters
    provider.client.messages.create.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.7,
        stop_sequences=["\nUser:"],
    )

    # Check if elapsed_time is a float (indicating the timer was used)
    assert isinstance(elapsed_time, float)

@patch("providers.anthropic_provider.anthropic.Anthropic")
def test_perform_inference_streaming(mock_anthropic_client_class, setup_anthropic_provider, capfd):
    """Test perform_inference_streaming handles streaming responses correctly."""
    provider = setup_anthropic_provider

    # Mock the messages.stream method to simulate a streaming response
    mock_client_instance = MagicMock()
    mock_stream = MagicMock()
    mock_stream.text_stream = iter(["chunk1", "chunk2", "chunk3"])
    mock_client_instance.messages.stream.return_value.__enter__.return_value = mock_stream
    provider.client = mock_client_instance  # Directly set the mock client

    # Call the method and capture the output with verbosity enabled
    provider.perform_inference_streaming("claude-3.5-sonnet", "Test prompt", max_output=100, verbosity=True)
    captured = capfd.readouterr()

    # Verify messages.stream is called with correct parameters
    provider.client.messages.stream.assert_called_once_with(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Test prompt"}],
        temperature=0.7,
        stop_sequences=["\nUser:"],
    )

    # Verify the output contains expected chunks and latency information
    assert "chunk1" in captured.out
    assert "chunk2" in captured.out
    assert "chunk3" in captured.out
    assert "Time to First Token" in captured.out
    assert "Total Response Time" in captured.out

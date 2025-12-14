import pytest
import os
from unittest.mock import patch, MagicMock
from providers.google_provider import GoogleGemini
import google.generativeai as genai


@pytest.fixture
def setup_google_gemini():
    """Fixture to set up and return an instance of GoogleGemini with a mocked API key."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key"}):
        return GoogleGemini()


def test_google_gemini_initialization(setup_google_gemini):
    """Test that GoogleGemini initializes correctly and sets the model_map."""
    provider = setup_google_gemini

    # Ensure model_map is set correctly
    assert provider.model_map == {
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
        "gemini-2.0-flash": "gemini-2.0-flash-001",
        "common-model": "gemini-2.0-flash-001",
    }


def test_google_gemini_api_key():
    """Test that GoogleGemini raises an error if GEMINI_API_KEY is missing."""
    with patch.dict(os.environ, {}, clear=True):  # Clear environment variables
        with pytest.raises(
            EnvironmentError, match="GEMINI_API_KEY is not set in the environment."
        ):
            GoogleGemini()


@patch("providers.google_provider.genai.GenerativeModel")
def test_perform_inference(mock_gen_model_class, setup_google_gemini):
    """Test perform_inference calls the correct methods with the correct parameters."""
    provider = setup_google_gemini

    # Mock the GenerativeModel instance and its generate_content method
    mock_gen_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_response.usage_metadata.candidates_token_count = 100
    mock_gen_model_instance.generate_content.return_value = mock_response
    mock_gen_model_class.return_value = mock_gen_model_instance

    # Call the method with max_output and verbosity enabled
    response = provider.perform_inference(
        "gemini-2.5-flash", "Test prompt", max_output=100, verbosity=True,
    )

    # Verify generate_content is called with correct parameters
    mock_gen_model_instance.generate_content.assert_called_once_with(
        "Test prompt",
        generation_config=genai.types.GenerationConfig(max_output_tokens=100),
    )

    # Ensure the response is a dict
    assert isinstance(response, dict)


@patch("providers.google_provider.genai.GenerativeModel")
def test_perform_inference_streaming(mock_gen_model_class, setup_google_gemini, capfd):
    """Test perform_inference_streaming handles streaming responses correctly."""
    provider = setup_google_gemini

    # Mock the GenerativeModel instance and simulate streaming response
    mock_gen_model_instance = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = [
        MagicMock(text="chunk1"),
        MagicMock(text="chunk2"),
        MagicMock(text="chunk3"),
    ]
    mock_gen_model_instance.generate_content.return_value = mock_stream
    mock_gen_model_class.return_value = mock_gen_model_instance

    # Call the method and capture the output with verbosity enabled
    response_list = provider.perform_inference_streaming(
        "gemini-2.5-flash", "Test prompt", max_output=100, verbosity=True,
    )
    captured = capfd.readouterr()

    # Verify generate_content is called with correct parameters for streaming
    mock_gen_model_instance.generate_content.assert_called_once_with(
        "Test prompt",
        generation_config=genai.types.GenerationConfig(max_output_tokens=100),
        stream=True,
    )

    # Verify the output contains expected chunks and latency information
    assert "chunk1" in captured.out
    assert "chunk2" in captured.out
    assert "chunk3" in captured.out
    assert "Time to First Token" in captured.out
    assert "Total Response Time" in captured.out

    # Ensure the response is a list
    assert isinstance(response_list, list)

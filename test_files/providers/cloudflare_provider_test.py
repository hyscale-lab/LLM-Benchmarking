import pytest
import os
from unittest.mock import patch, MagicMock
from providers.cloudflare_provider import Cloudflare


@pytest.fixture
def setup_cloudflare_provider():
    """Fixture to set up and return an instance of Cloudflare with mocked environment variables."""
    with patch.dict(
        os.environ,
        {
            "CLOUDFLARE_ACCOUNT_ID": "test_account_id",
            "CLOUDFLARE_AI_TOKEN": "test_api_token",
        },
    ):
        return Cloudflare()


def test_cloudflare_provider_initialization(setup_cloudflare_provider):
    """Test that Cloudflare initializes correctly with the environment variables."""
    provider = setup_cloudflare_provider

    # Ensure the model_map is set correctly
    assert provider.model_map == {
        "google-gemma-2b-it": "@cf/google/gemma-2b-it-lora",
        "phi-2": "@cf/microsoft/phi-2",
        "meta-llama-3.2-3b-instruct": "@cf/meta/llama-3.2-3b-instruct",
        "mistral-7b-instruct-v0.1": "@cf/mistral/mistral-7b-instruct-v0.1",
        "meta-llama-3.1-70b-instruct": "@cf/meta/llama-3.1-70b-instruct",
        "meta-llama-2-7b":"@cf/meta/llama-2-7b-chat-fp16",
        "deepseek-ai-R1-32b": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "common-model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    }


def test_cloudflare_provider_missing_env_variables():
    """Test that Cloudflare raises an error if environment variables are missing."""
    with patch.dict(os.environ, {}, clear=True):  # Clear environment variables
        with pytest.raises(
            ValueError,
            match="Cloudflare account ID and API token must be provided either as arguments or environment variables.",
        ):
            Cloudflare()


@patch("providers.cloudflare_provider.requests.post")
def test_perform_inference(mock_post, setup_cloudflare_provider):
    """Test perform_inference method."""
    provider = setup_cloudflare_provider
    mock_response = MagicMock()
    mock_response.json.return_value = {"result": {"response": "Test response"}}
    mock_post.return_value = mock_response

    elapsed_time = provider.perform_inference(
        "google-gemma-2b-it", "Test prompt", max_output=100
    )

    # Verify that the request was made with the correct parameters
    mock_post.assert_called_once_with(
        f"https://api.cloudflare.com/client/v4/accounts/test_account_id/ai/run/@cf/google/gemma-2b-it-lora",
        headers={"Authorization": "Bearer test_api_token"},
        json={
            "messages": [
                {"role": "system", "content": provider.system_prompt},
                {"role": "user", "content": "Test prompt"},
            ],
            "max_tokens": 100,
        },
        timeout=500,
    )

    # Ensure the response time is a float
    assert isinstance(elapsed_time, float)


@patch("providers.cloudflare_provider.requests.post")
def test_perform_inference_streaming(mock_post, setup_cloudflare_provider, capfd):
    """Test perform_inference_streaming method handles streaming responses."""
    provider = setup_cloudflare_provider
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = [
        b'data: {"response":"chunk1"}',
        b'data: {"response":"chunk2"}',
        b'data: {"response":"chunk3"}',
        b"data: [DONE]",
    ]
    mock_post.return_value = mock_response

    provider.perform_inference_streaming(
        "meta-llama-3.2-3b-instruct", "Test prompt", max_output=100
    )
    captured = capfd.readouterr()

    # Verify the request was made with the correct parameters
    mock_post.assert_called_once_with(
        f"https://api.cloudflare.com/client/v4/accounts/test_account_id/ai/run/@cf/meta/llama-3.2-3b-instruct",
        headers={
            "Authorization": "Bearer test_api_token",
            "Content-Type": "application/json",
        },
        json={
            "stream": True,
            "messages": [
                {"role": "system", "content": provider.system_prompt},
                {"role": "user", "content": "Test prompt"},
            ],
            "max_tokens": 100,
        },
        stream=True,
        timeout=500,
    )

    # Verify the output contains expected tokens
    assert "chunk1" in captured.out
    assert "chunk2" in captured.out
    assert "chunk3" in captured.out

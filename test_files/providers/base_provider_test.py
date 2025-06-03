import pytest
from unittest.mock import MagicMock, patch
from providers import BaseProvider


@pytest.fixture
def base_provider():
    mock_client = MagicMock()
    api_key = "test_api_key"
    provider = BaseProvider(api_key=api_key, client_class=mock_client)
    provider.model_map = {"test-model": "model_id_test"}
    provider.system_prompt = "This is a test system prompt."
    provider.max_tokens = 100
    return provider


@patch("providers.base_provider.timer", side_effect=[0, 1.0])
@patch.object(BaseProvider, "log_metrics")
@patch.object(BaseProvider, "display_response")
def test_perform_inference(
    mock_display_response, mock_log_metrics, mock_timer, base_provider
):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response content."
    base_provider.client.chat.completions.create.return_value = mock_response

    # Call perform_inference
    elapsed_time = base_provider.perform_inference(
        "test-model", "What is the test prompt?"
    )

    base_provider.client.chat.completions.create.assert_called_once_with(
        model="model_id_test",
        messages=[
            {"role": "system", "content": "This is a test system prompt."},
            {"role": "user", "content": "What is the test prompt?"},
        ],
        max_tokens=100,
        timeout=(10,100)
    )

    mock_log_metrics.assert_called_with("test-model", "response_times", 1.0)
    mock_display_response.assert_called_with(mock_response, 1.0)
    assert elapsed_time == 1.0


@patch("providers.base_provider.timer", side_effect=[0, 0.5, 1.0, 1.5, 2.0])
@patch.object(BaseProvider, "log_metrics")
@patch.object(BaseProvider, "display_response")
def test_perform_inference_streaming(
    mock_display_response, mock_log_metrics, mock_timer, base_provider
):
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Test chunk 1 content."
    mock_chunk1.choices[0].finish_reason = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = "Test chunk 2 content."
    mock_chunk2.choices[0].finish_reason = None

    mock_chunk3 = MagicMock()
    mock_chunk3.choices = [MagicMock()]
    mock_chunk3.choices[0].delta.content = None
    mock_chunk3.choices[0].finish_reason = "stop"

    # Simulate the stream response
    base_provider.client.chat.completions.create.return_value = [
        mock_chunk1,
        mock_chunk2,
        mock_chunk3,
    ]

    # Call perform_inference_streaming
    base_provider.perform_inference_streaming(
        "test-model", "What is the test streaming prompt?"
    )

    base_provider.client.chat.completions.create.assert_called_once_with(
        model="model_id_test",
        messages=[
            {"role": "system", "content": "This is a test system prompt."},
            {"role": "user", "content": "What is the test streaming prompt?"},
        ],
        stream=True,
        max_tokens=100,
        timeout=(10,100)
    )

    avg_tbt = sum([0.5, 0.5]) / len([0.5, 0.5])
    mock_log_metrics.assert_any_call("test-model", "timetofirsttoken", 0.5)
    mock_log_metrics.assert_any_call("test-model", "response_times", 2.0)
    mock_log_metrics.assert_any_call("test-model", "timebetweentokens", avg_tbt)

    mock_display_response.assert_not_called()

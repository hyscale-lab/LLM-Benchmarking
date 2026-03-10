import os
import mimetypes
from timeit import default_timer as timer
from google import genai
from google.genai import types
from providers.provider_interface import ProviderInterface


class GoogleGemini(ProviderInterface):
    def __init__(self):
        """
        Initializes the GoogleGeminiProvider with model mapping and API configuration.
        """
        super().__init__()

        # Map of model names to specific Google Gemini model identifiers
        self.model_map = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-2.0-flash": "gemini-2.0-flash-001",
            "common-model": "gemini-2.0-flash-001",
            "cache-model": "gemini-2.5-flash",
            "vision-model-01": "meta/llama-4-maverick-17b-128e-instruct-maas",
        }

        # for multiturn caching
        self._google_cache_name = None

    def initialize_client(self):
        # Configure API key for Google Gemini
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise EnvironmentError("Google cloud credentials not set.")

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        llama_4_location = os.getenv("LLAMA_4_GOOGLE_CLOUD_LOCATION")
        if not project:
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT is not set in the environment.")
        if not location:
            raise EnvironmentError("GOOGLE_CLOUD_LOCATION is not set in the environment.")
        if not llama_4_location:
            raise EnvironmentError("LLAMA_4_GOOGLE_CLOUD_LOCATION is not set in the environment.")

        self._clients = {
            'llama-4-maverick': genai.Client(
                vertexai=True,
                project=project,
                location=llama_4_location
            ),
            'default': genai.Client(
                vertexai=True,
                project=project,
                location=location
            ),
        }

    def _set_client_by_model(self, model_id):
        if 'llama-4-maverick' in model_id:
            self._client = self._clients['llama-4-maverick']
        else:
            self._client = self._clients['default']

    def get_model_name(self, model):
        """
        Retrieves the model ID from the model_map.
        """
        return self.model_map.get(model)

    def normalize_messages(self, messages):
        if isinstance(messages, str):
            normalized_msgs = messages
        elif isinstance(messages, list):
            normalized_msgs = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "user":
                    parts = []
                    if isinstance(content, str):
                        parts.append(types.Part.from_text(text=content))
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                parts.append(types.Part.from_text(text=item["text"]))
                            elif item.get("type") == "image":
                                path = item["image_path"]

                                # Process image
                                mime_type, _ = mimetypes.guess_type(path)
                                mime_type = mime_type or "image/jpeg"
                                with open(path, "rb") as f:
                                    image_bytes = f.read()

                                parts.append(types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=mime_type
                                ))
                            else:
                                print(f"Invalid content item type: {item.get('type')}")

                    normalized_msgs.append(
                        types.Content(role="user", parts=parts)
                    )
                elif role == "assistant":
                    normalized_msgs.append(
                        types.Content(role="model", parts=[types.Part.from_text(text=content)])
                    )
                else:
                    print(f"Invalid role found in messages: {role}")

        return normalized_msgs

    def apply_cache_markers(self, messages):
        if not isinstance(messages, list) or len(messages) < 3:
            # Turn 1: no history to cache yet; reset for new conversation
            self._google_cache_name = None
            return messages

        # Cache everything except the last user message
        history_to_cache = messages[:-1]
        tail_messages = [messages[-1]]

        model_id = self.model_map.get("cache-model")
        self._set_client_by_model(model_id)
        normalized_history = self.normalize_messages(history_to_cache)
        try:
            cache = self._client.caches.create(
                model=model_id,
                config=types.CreateCachedContentConfig(
                    contents=normalized_history,
                    system_instruction=self.system_prompt,
                    ttl="300s",
                )
            )
            self._google_cache_name = cache.name
            print(f"[Google Cache] Created: {cache.name}")
            return tail_messages
        except Exception as e:
            print(f"[WARN] Google cache creation failed: {e}. Falling back to full context.")
            self._google_cache_name = None
            return messages

    def construct_text_response(self, raw_response):
        if isinstance(raw_response, dict):
            text_response = raw_response['candidates'][0]['content']['parts'][0]['text']
        elif isinstance(raw_response, list):
            text_response = "".join(
                block['candidates'][0]['content']['parts'][0]['text']
                for block in raw_response
            )

        return text_response

    def get_response_usage(self, response, streaming):
        if not response:
            return {"total_input": 0, "output": 0}

        usage = response[-1].get('usage_metadata', {}) if streaming else response.get('usage_metadata', {})
        result = {
            "total_input": usage.get('prompt_token_count', 0),
            "output": usage.get('candidates_token_count', 0)
        }
        cached = usage.get('cached_content_token_count')
        if cached:
            result["cache_read"] = cached
        return result

    def perform_inference(self, model, messages, max_output=100, verbosity=True):
        """
        Performs inference on a single prompt and returns the time taken for response generation.
        """
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} is not supported by GoogleGeminiProvider.")

            self._set_client_by_model(model_id)

            start_time = timer()
            response = self._client.models.generate_content(
                model=model_id,
                contents=self.normalize_messages(messages),
                config=types.GenerateContentConfig(
                    cached_content=self._google_cache_name,
                    max_output_tokens=max_output,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ) if self._google_cache_name else types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    max_output_tokens=max_output,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            elapsed = timer() - start_time

            usage = getattr(response, "usage_metadata", None)
            total_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0

            tbt = elapsed / max(total_tokens, 1)
            tps = total_tokens / elapsed

            self.log_metrics(model, "response_times", elapsed)
            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)

            if verbosity:
                print(f"Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                print(response.text)
                print(f"\nGenerated in {elapsed:.2f} seconds")
            return response.model_dump()

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return e

    def perform_inference_streaming(
        self, model, messages, max_output=100, verbosity=True
    ):
        """
        Performs streaming inference on a single prompt, capturing latency metrics and output.
        """
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} is not supported by GoogleGeminiProvider.")

            self._set_client_by_model(model_id)

            start_time = timer()
            response = self._client.models.generate_content_stream(
                model=model_id,
                contents=self.normalize_messages(messages),
                config=types.GenerateContentConfig(
                    cached_content=self._google_cache_name,
                    max_output_tokens=max_output,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ) if self._google_cache_name else types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    max_output_tokens=max_output,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )

            ttft = None
            first_token_time = None
            streamed_output = []
            total_tokens = 0
            first_chunk_tokens = 0

            response_list = []
            for chunk in response:
                response_list.append(chunk.model_dump())
                current_time = timer()
                text = getattr(chunk, "text", "") or ""

                # Estimate the number of tokens in the current chunk
                num_tokens = 0

                if timer() - start_time > 90:
                    print("[WARN] Streaming exceeded 90s, stopping early.")
                    break
                if text:
                    try:
                        num_tokens = int(self._client.models.count_tokens(model=model_id, contents=text).total_tokens)
                        if num_tokens <= 0:
                            num_tokens = 1
                    except Exception:
                        num_tokens = 1

                if first_token_time is None and text:
                    first_token_time = current_time
                    ttft = first_token_time - start_time
                    first_chunk_tokens = max(num_tokens, 0)
                    if verbosity:
                        print(f"Time to First Token (TTFT): {ttft:.4f} seconds")

                # Calculate inter-token latency per token in the chunk
                if num_tokens > 0:
                    total_tokens += num_tokens

                if verbosity and text:
                    print(text, end="", flush=True)
                streamed_output.append(text)

            total_time = timer() - start_time
            if ttft is None:
                ttft = total_time
            non_first_latency = max(total_time - ttft, 0.0)
            subsequent_tokens = max(total_tokens - first_chunk_tokens, 0)
            avg_tbt = (non_first_latency / subsequent_tokens) if subsequent_tokens > 0 else 0.0

            if verbosity:
                print(f"\nTotal Response Time: {total_time:.4f} seconds")
                print(f"total tokens {total_tokens}")
                print(f"Avg TBT: {avg_tbt:.4f} seconds")

            self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(
                model, "tps", total_tokens / total_time if total_time > 0 else 0
            )

            return response_list

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return e

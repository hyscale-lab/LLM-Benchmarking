import os
from timeit import default_timer as timer
import google.generativeai as genai
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
        }

        # Configure API key for Google Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set in the environment.")

        genai.configure(api_key=api_key)
        self.model = None

    def get_model_name(self, model):
        """
        Retrieves the model ID from the model_map.
        """
        return self.model_map.get(model)

    def _initialize_model(self, model_id):
        """
        Initializes the generative model instance for the specified model_id.
        """
        self.model = genai.GenerativeModel(model_id)

    def perform_inference(self, model, prompt, max_output=100, verbosity=True):
        """
        Performs inference on a single prompt and returns the time taken for response generation.
        """
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} is not supported by GoogleGeminiProvider.")

            self._initialize_model(model_id)

            start_time = timer()
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output
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
            return response.to_dict()

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return e

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        """
        Performs streaming inference on a single prompt, capturing latency metrics and output.
        """
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} is not supported by GoogleGeminiProvider.")

            self._initialize_model(model_id)

            start_time = timer()
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_output
                ),
                stream=True,
            )

            ttft = None
            first_token_time = None
            streamed_output = []
            total_tokens = 0
            first_chunk_tokens = 0

            response_list = []
            for chunk in response:
                response_list.append(chunk.to_dict())
                current_time = timer()
                text = getattr(chunk, "text", "") or ""

                # Estimate the number of tokens in the current chunk
                num_tokens = 0

                if timer() - start_time > 90:
                    print("[WARN] Streaming exceeded 90s, stopping early.")
                    break
                if text:
                    try:
                        num_tokens = int(self.model.count_tokens(text).total_tokens)
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

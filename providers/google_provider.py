import os
from providers.provider_interface import ProviderInterface
import google.generativeai as genai
from timeit import default_timer as timer
import numpy as np
import asyncio


class GoogleGemini(ProviderInterface):
    def __init__(self):
        """
        Initializes the GoogleGeminiProvider with model mapping and API configuration.
        """
        super().__init__()

        # Map of model names to specific Google Gemini model identifiers
        self.model_map = {
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-1.5-flash-8b": "gemini-1.5-flash-8b",
            "gemini-1.5-pro": "gemini-1.5-pro",
            "common-model": "gemini-1.5-flash",
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
            tps = (total_tokens / elapsed)

            self.log_metrics(model, "response_times", elapsed)
            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)

            if verbosity:
                print(f"Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                print(response.text)
                print(f"\nGenerated in {elapsed:.2f} seconds")
            return elapsed
        
        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None, None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        """
        Performs streaming inference on a single prompt, capturing latency metrics and output.
        """
        model_id = self.get_model_name(model)
        if model_id is None:
            raise ValueError(f"Model {model} is not supported by GoogleGeminiProvider.")

        self._initialize_model(model_id)

        inter_token_latencies = []
        start_time = timer()
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output
            ),
            stream=True,
        )

        first_token_time = None
        prev_token_time = start_time
        streamed_output = []
        total_tokens = 0

        for chunk in response:
            current_time = timer()

            if first_token_time is None:
                first_token_time = current_time
                TTFT = first_token_time - start_time
                prev_token_time = first_token_time
                if verbosity:
                    print(f"Time to First Token (TTFT): {TTFT:.4f} seconds")

            # Estimate the number of tokens in the current chunk
            num_tokens = int(self.model.count_tokens(chunk.text).total_tokens)
            total_tokens += num_tokens

            # Calculate inter-token latency per token in the chunk
            if num_tokens > 0:
                inter_token_latency = (current_time - prev_token_time) / num_tokens
                for _ in range(num_tokens):
                    inter_token_latencies.append(inter_token_latency)

            prev_token_time = current_time
            if verbosity and chunk.text:
                print(chunk.text, end="", flush=True)
            streamed_output.append(chunk.text)

        total_time = timer() - start_time
        if verbosity:
            print(f"\nTotal Response Time: {total_time:.4f} seconds")
            print(f"total tokens {len(inter_token_latencies)}")

        avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
        self.log_metrics(model, "timetofirsttoken", TTFT)
        self.log_metrics(model, "response_times", total_time)
        self.log_metrics(model, "timebetweentokens", avg_tbt)

        # Calculate additional latency metrics
        median_latency = (
            np.median(inter_token_latencies) if inter_token_latencies else 0
        )
        p95_latency = (
            np.percentile(inter_token_latencies, 95) if inter_token_latencies else 0
        )

        self.log_metrics(model, "timebetweentokens_median", median_latency)
        self.log_metrics(model, "timebetweentokens_p95", p95_latency)
        self.log_metrics(model, "totaltokens", total_tokens)
        self.log_metrics(
            model, "tps", total_tokens / total_time if total_time > 0 else 0
        )

        return streamed_output
    
    def perform_trace_mode(self, proxy_server, load_generator, num_requests, verbosity):
        # Set handler for proxy
        async def data_handler(data, streaming):
            if streaming:
                print("\nRequest not sent. Streaming not allowed in trace mode.")
                return [{"error": "Streaming not allowed in trace mode."}]
            
            def inference_sync():
                try:
                    model_id = data.pop('model')
                    if not model_id or model_id not in self.model_map.values():
                        raise Exception(f"Model {model_id} not found in model map.")
                    model = next((k for k, v in self.model_map.items() if v == model_id))
                    self._initialize_model(model_id)
                                        
                    # Non-streaming inference
                    start_time = timer()
                    response = self.model.generate_content(
                        contents=data.pop('contents'),
                        stream=data.pop('stream'),
                        generation_config=data
                    )
                    elapsed_time = timer() - start_time

                    usage = getattr(response, "usage_metadata", None)
                    total_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
                    tbt = elapsed_time / max(total_tokens, 1)
                    tps = (total_tokens / elapsed_time)
                    self.log_metrics(model, "response_times", elapsed_time)
                    self.log_metrics(model, "totaltokens", total_tokens)
                    self.log_metrics(model, "timebetweentokens", tbt)
                    self.log_metrics(model, "tps", tps)

                    if verbosity:
                        print()
                        print(f"##### Generated in {elapsed_time:.2f} seconds")
                        print(f"##### Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                        print(f"Response: {response.text}")
                        
                    return response.to_dict()
                
                except Exception as e:
                    print(f"\nInference failed: {e}")
                    return [{"error": f"Inference failed: {e}"}] if streaming else {"error": f"Inference failed: {e}"}
            
            response = await asyncio.to_thread(inference_sync)            
            return response
        
        proxy_server.set_handler(data_handler)

        # Start load generator
        load_generator.send_loads(
            self.trace_dataset_path,
            self.trace_result_path,
            sampling_rate=100,
            recur_step=10,
            limit=num_requests,
            max_drift=100,
            upscale='ars'
        )

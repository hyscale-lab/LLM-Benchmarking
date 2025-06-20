import os
from providers.provider_interface import ProviderInterface
import google.generativeai as genai
from timeit import default_timer as timer
import numpy as np
import math


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
            "common-model-small": "gemini-2.0-flash-lite"
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

            self.log_metrics(model, "response_times", elapsed)
            if verbosity:
                print(response.text)
                print(f"\nGenerated in {elapsed:.2f} seconds")
            return elapsed
        
        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None, None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True, correct_answer=None
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
        generated_text = ""
        for chunk in response:
            current_time = timer()

            if first_token_time is None:
                first_token_time = current_time
                ttft = first_token_time - start_time
                prev_token_time = first_token_time
                if verbosity:
                    print(f"Time to First Token (TTFT): {ttft:.4f} seconds")

            # Estimate the number of tokens in the current chunk
            num_tokens = int(self.model.count_tokens(chunk.text).total_tokens)
            total_tokens += num_tokens

            # Calculate inter-token latency per token in the chunk
            if num_tokens > 0:
                inter_token_latency = (current_time - prev_token_time) / num_tokens
                for _ in range(num_tokens):
                    inter_token_latencies.append(inter_token_latency)

            prev_token_time = current_time
            # print(chunk.text)
            if verbosity and chunk.text:
                print(chunk.text, end="", flush=True)
            streamed_output.append(chunk.text)

        total_time = timer() - start_time
        if verbosity:
            print(f"\nTotal Response Time: {total_time:.4f} seconds")
            print(f"total tokens {len(inter_token_latencies)}")

        # avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
        # self.log_metrics(model, "timetofirsttoken", TTFT)
        # self.log_metrics(model, "response_times", total_time)
        # self.log_metrics(model, "timebetweentokens", avg_tbt)

        # # Calculate additional latency metrics
        # median_latency = (
        #     np.median(inter_token_latencies) if inter_token_latencies else 0
        # )
        # p95_latency = (
        #     np.percentile(inter_token_latencies, 95) if inter_token_latencies else 0
        # )

        # self.log_metrics(model, "timebetweentokens_median", median_latency)
        # self.log_metrics(model, "timebetweentokens_p95", p95_latency)
        # self.log_metrics(model, "totaltokens", total_tokens)
        # self.log_metrics(
        #     model, "tps", total_tokens / total_time if total_time > 0 else 0
        # )

        avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
        if verbosity:

            print(
                f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Avg TBT: {avg_tbt:.4f}, Time to First Token (TTFT): {ttft:.4f} seconds, Total Response Time: {total_time:.4f} seconds"
            )
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timetofirsttoken", ttft)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "response_times", total_time)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens", inter_token_latencies)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_avg", avg_tbt)        
        median = np.percentile(inter_token_latencies, 50)
        p95 = np.percentile(inter_token_latencies, 95)
        p99 = np.percentile(inter_token_latencies, 99)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_median", median)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_p95", p95)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_p99", p99)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "totaltokens", len(inter_token_latencies) + 1)
        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "tps", (len(inter_token_latencies) + 1) / total_time)
        self.log_metrics(
            model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "dpsk_output", streamed_output
        )

        return streamed_output

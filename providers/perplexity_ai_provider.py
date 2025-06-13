import os
from timeit import default_timer as timer
import numpy as np
from openai import OpenAI
from providers.base_provider import BaseProvider
import math

class PerplexityAI(BaseProvider):
    """perplexity provider class"""

    def __init__(self):
        """
        Initializes the AnthropicProvider with the necessary API key and client.
        """
        perplexity_api = os.environ.get("PERPLEXITY_AI_API")

        if not perplexity_api:
            raise ValueError(
                "Perplexity AI API token must be provided as an environment variable."
            )

        client_class = OpenAI
        base_url = "https://api.perplexity.ai"

        super().__init__(
            api_key=perplexity_api, client_class=client_class, base_url=base_url
        )
        # model names mapping
        self.model_map = {
            "sonar": "sonar",
            "sonar-pro": "sonar-pro",
            "sonar-reasoning-pro": "sonar-reasoning-pro",
            "common-model": "sonar-pro",
            "common-model-small": "sonar"
        }

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for provider.")
            first_token_time = None
            inter_token_latencies = []
            total_tokens = 0

            start = timer()
            print("HELLO")
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                max_tokens=max_output,
                timeout=500
            )
            previous_completion_tokens = 0  # Initialize previous token count
            print("here?")
            print(response)
            for chunk in response:
                print(chunk)
                current_completion_tokens = chunk.usage.completion_tokens
                new_tokens = current_completion_tokens - previous_completion_tokens
                previous_completion_tokens = current_completion_tokens
                if first_token_time is None:
                    first_token_time = timer()
                    ttft = first_token_time - start
                    prev_token_time = first_token_time
                    if verbosity:
                        print(f"\nTime to First Token (TTFT): {ttft:.4f} seconds\n")

                if chunk.choices[0].finish_reason:
                    elapsed = timer() - start
                    if verbosity:
                        print(f"\nTotal Response Time: {elapsed:.4f} seconds")
                    break

                time_to_next_token = timer()
                inter_token_latency = (time_to_next_token - prev_token_time) / max(
                    1, new_tokens
                )
                prev_token_time = time_to_next_token
                for _ in range(new_tokens):
                    inter_token_latencies.append(inter_token_latency)

                total_tokens += new_tokens

                print(chunk.choices[0].delta.content or "", end="")
                # Print the content of the chunk if verbosity is enabled
                # if verbosity:
                #     if len(inter_token_latencies) < 20:
                #         print(chunk.choices[0].delta.content or "", end="", flush=True)
                #     elif len(inter_token_latencies) == 20:
                #         print("...")

            if verbosity:
                print(
                    f"\nNumber of output tokens: {total_tokens}, "
                    f"Time to First Token (TTFT): {ttft:.4f} seconds, "
                    f"Total Response Time: {elapsed:.4f} seconds"
                )
            # # Log metrics
            # self.log_metrics(model, "timetofirsttoken", ttft)
            # self.log_metrics(model, "response_times", elapsed)
            # avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            # self.log_metrics(model, "timebetweentokens", avg_tbt)

            # # Calculate additional latency metrics
            # median = np.median(inter_token_latencies) if inter_token_latencies else 0
            # p95 = np.percentile(inter_token_latencies, 95) if inter_token_latencies else 0
            # self.log_metrics(model, "timebetweentokens_median", median)
            # self.log_metrics(model, "timebetweentokens_p95", p95)
            # self.log_metrics(model, "totaltokens", total_tokens)
            # self.log_metrics(model, "tps", total_tokens / elapsed if elapsed > 0 else 0)

            avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            if verbosity:

                print(
                    f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Avg TBT: {avg_tbt:.4f}, Time to First Token (TTFT): {ttft:.4f} seconds, Total Response Time: {elapsed:.4f} seconds"
                )
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timetofirsttoken", ttft)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "response_times", elapsed)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens", avg_tbt)
            median = np.percentile(inter_token_latencies, 50)
            p95 = np.percentile(inter_token_latencies, 95)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_median", median)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_p95", p95)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "totaltokens", len(inter_token_latencies) + 1)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "tps", (len(inter_token_latencies) + 1) / elapsed)

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return None, None

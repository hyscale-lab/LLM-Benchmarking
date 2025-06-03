import os
import time
import requests
import numpy as np
from timeit import default_timer as timer
from providers.provider_interface import ProviderInterface

# from IPython.display import display, Image, Markdown, Audio
# import logging


class Cloudflare(ProviderInterface):
    def __init__(self):
        """
        Initializes the Cloudflare with the necessary API key and client.
        """
        super().__init__()

        cloudflare_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        cloudflare_api_token = os.environ.get("CLOUDFLARE_AI_TOKEN")

        if not cloudflare_account_id or not cloudflare_api_token:
            raise ValueError(
                "Cloudflare account ID and API token must be provided either as arguments or environment variables."
            )

        self.cloudflare_account_id = cloudflare_account_id
        self.cloudflare_api_token = cloudflare_api_token

        # model names
        self.model_map = {
            "google-gemma-2b-it": "@cf/google/gemma-2b-it-lora",
            "phi-2": "@cf/microsoft/phi-2",
            "meta-llama-3.2-3b-instruct": "@cf/meta/llama-3.2-3b-instruct",
            "mistral-7b-instruct-v0.1": "@cf/mistral/mistral-7b-instruct-v0.1",
            "meta-llama-3.1-70b-instruct": "@cf/meta/llama-3.1-70b-instruct",
            "meta-llama-2-7b": "@cf/meta/llama-2-7b-chat-fp16",
            "deepseek-ai-R1-32b": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
            "common-model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        }

    def get_model_name(self, model):
        return self.model_map.get(model, None)  # or model

    def perform_inference(self, model, prompt, max_output=100, verbosity=True):
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                print(f"Model {model} not available for provider {model_id}")
            start_time = timer()
            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/run/{model_id}",
                headers={"Authorization": f"Bearer {self.cloudflare_api_token}"},
                json={
                    "messages": [
                        # {"role": "system", "content": "Explain your answer step-by-step."},
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_output,  # self.max_tokens
                },
                timeout=500,
            )

            elapsed = timer() - start_time
            # print("request sucess")
            # log response times metric
            self.log_metrics(model, "response_times", elapsed)

            inference = response.json()
            print(inference)
            # logging.debug(inference["result"]["response"])
            if verbosity:
                print(inference["result"]["response"][:50])

                print(f"#### _Generated in *{elapsed:.2f}* seconds_")
            return elapsed

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None, None
            
    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):

        try:
            inter_token_latencies = []
            model_id = self.get_model_name(model)
            start_time = time.perf_counter()

            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/run/{model_id}",
                headers={
                    "Authorization": f"Bearer {self.cloudflare_api_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_output,
                },
                stream=True,
                timeout=500,
            )

            first_token_time = None
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft = first_token_time - start_time
                        prev_token_time = first_token_time
                        if verbosity:
                            print(f"##### Time to First Token (TTFT): {ttft:.4f} seconds\n")

                    line_str = line.decode("utf-8").strip()

                    # Check if the stream is done
                    if line_str == "data: [DONE]":
                        end_time = time.perf_counter()
                        total_time = end_time - start_time
                        if verbosity:
                            print(f"##### Total Response Time: {total_time:.4f} seconds")
                        break
                    time_to_next_token = time.perf_counter()
                    inter_token_latency = time_to_next_token - prev_token_time
                    prev_token_time = time_to_next_token

                    inter_token_latencies.append(inter_token_latency)
                    print(line_str[19:].split('"')[0], end='')

            if verbosity:
                print(
                    f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Time to First Token (TTFT): {ttft:.4f} seconds, Total Response Time: {total_time:.4f} seconds"
                )

            avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            print(f"\nAvg TBT: {avg_tbt:.4f}, {len(inter_token_latencies)}")
            self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            median = np.percentile(inter_token_latencies, 50)
            p95 = np.percentile(inter_token_latencies, 95)
            self.log_metrics(model, "timebetweentokens_median", median)
            self.log_metrics(model, "timebetweentokens_p95", p95)
            self.log_metrics(model, "totaltokens", len(inter_token_latencies) + 1)
            self.log_metrics(model, "tps", (len(inter_token_latencies) + 1) / total_time)
        
        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return None, None

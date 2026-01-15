import os
import time
from timeit import default_timer as timer
import json
import requests
from providers.provider_interface import ProviderInterface
from utils.accuracy_mixin import AccuracyMixin

# from IPython.display import display, Image, Markdown, Audio
# import logging


class Cloudflare(AccuracyMixin, ProviderInterface):
    def __init__(self):
        """
        Initializes the Cloudflare with the necessary API key and client.
        """
        super().__init__()

        # model names
        self.model_map = {
            "google-gemma-2b-it": "@cf/google/gemma-2b-it-lora",
            "phi-2": "@cf/microsoft/phi-2",
            "meta-llama-3.2-3b-instruct": "@cf/meta/llama-3.2-3b-instruct",
            "mistral-7b-instruct-v0.1": "@cf/mistral/mistral-7b-instruct-v0.1",
            "meta-llama-3.1-70b-instruct": "@cf/meta/llama-3.1-70b-instruct",
            "common-model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "reasoning-model": ["@cf/openai/gpt-oss-120b"]
        }

    def initialize_client(self):
        cloudflare_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        cloudflare_api_token = os.environ.get("CLOUDFLARE_AI_TOKEN")

        if not cloudflare_account_id or not cloudflare_api_token:
            raise ValueError(
                "Cloudflare account ID and API token must be provided either as arguments or environment variables."
            )
        
        self.cloudflare_account_id = cloudflare_account_id
        self.cloudflare_api_token = cloudflare_api_token

    def get_model_name(self, model):
        return self.model_map.get(model, None)  # or model

    def normalize_messages(self, messages):
        if isinstance(messages, str):
            normalized_msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": messages}
            ]
        elif isinstance(messages, list):
            normalized_msgs = [{"role": "system", "content": self.system_prompt}]
            for msg in messages:
                role = msg["role"]

                if role in ["user", "assistant"]:
                    normalized_msgs.append(msg)
                else:
                    print(f"Invalid role found in messages: {role}")

        return normalized_msgs
    
    def construct_text_response(self, raw_response):
        if isinstance(raw_response, dict):
            text_response = raw_response["result"]["response"]
        elif isinstance(raw_response, list):
            text_response = "".join(
                str(block['response'])
                for block in raw_response
                if isinstance(block, dict) and 'response' in block
            )

        return text_response

    def perform_inference(self, model, messages, max_output=100, verbosity=True):
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                print(f"Model {model} not available for provider {model_id}")
            start_time = timer()
            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/run/{model_id}",
                headers={"Authorization": f"Bearer {self.cloudflare_api_token}"},
                json={
                    "messages": self.normalize_messages(messages),
                    "max_tokens": max_output,  # self.max_tokens
                },
                timeout=500,
            )

            elapsed = timer() - start_time

            inference = response.json()

            meta = inference.get("result", {})
            usage = meta.get("usage", {})
            total_tokens = usage.get("completion_tokens") or 0

            tbt = elapsed / max(total_tokens, 1)
            tps = (total_tokens / elapsed)

            self.log_metrics(model, "response_times", elapsed)
            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)

            print(inference)
            # logging.debug(inference["result"]["response"])
            if verbosity:
                print(f"Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                print(inference["result"]["response"][:50])

                print(f"#### _Generated in *{elapsed:.2f}* seconds_")
            return inference

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return e

    def perform_inference_streaming(
        self, model, messages, max_output=100, verbosity=True
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
                    "messages": self.normalize_messages(messages),
                    "max_tokens": max_output,
                },
                stream=True,
                timeout=500,
            )

            ttft = None
            first_token_time = None
            prev_token_time = None

            response_list = []
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
                        response_list.append(line_str[6:])
                        end_time = time.perf_counter()
                        total_time = end_time - start_time
                        if verbosity:
                            print(f"##### Total Response Time: {total_time:.4f} seconds")
                        break
                    else:
                        response_list.append(json.loads(line_str[6:]))

                    time_to_next_token = time.perf_counter()
                    inter_token_latency = time_to_next_token - prev_token_time
                    prev_token_time = time_to_next_token

                    inter_token_latencies.append(inter_token_latency)
                    print(line_str[19:].split('"')[0], end='')

            total_time = time.perf_counter() - start_time
            token_count = (len(inter_token_latencies) + 1) if ttft is not None else 0
            non_first_latency = max(total_time - (ttft or 0.0), 0.0)
            avg_tbt = (non_first_latency / (token_count)) if token_count > 0 else 0.0

            if verbosity:
                print(
                    f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Time to First Token (TTFT): {ttft:.4f} seconds, Total Response Time: {total_time:.4f} seconds"
                )

            print(f"\nAvg TBT: {avg_tbt:.4f}, {len(inter_token_latencies)}")
            if ttft is not None:
                self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            self.log_metrics(model, "totaltokens", token_count)
            self.log_metrics(model, "tps", (token_count / total_time) if total_time > 0 else 0.0)
            return response_list

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return e

    def _chat_for_eval(self, model_id, messages):
        if model_id is None:
            return "", 0, 0.0

        start = timer()
        try:
            r = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.cloudflare_api_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "input": messages,
                    "max_tokens": 10000,
                    "temperature": 0.0,
                },
                timeout=500,
            )
            elapsed = timer() - start

            if r.status_code != 200:
                return "", 0, float(elapsed)

            data = r.json() or {}
            # print(data)
            result = data or {}

            outputs = result.get("output", [])
            text_parts = []

            for item in outputs:
                # Each item may have "content" list with multiple text entries
                for content in item.get("content", []):
                    if "text" in content:
                        text_parts.append(content["text"])

            # Join all text segments into one output string
            text = "\n".join(text_parts)

            usage = result.get("usage", {}) or {}
            tokens = (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or 0
            )

            # print(text)

            return text, int(tokens or 0), float(elapsed)

        except Exception as e:
            elapsed = timer() - start
            print(f"[ERROR] _chat_for_eval failed (Cloudflare): {e}")
            return "", 0, float(elapsed)

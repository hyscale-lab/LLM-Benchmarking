# base_provider.py for chat completions api
from timeit import default_timer as timer
import time
import numpy as np
from providers.provider_interface import ProviderInterface
import math

class BaseProvider(ProviderInterface):
    def __init__(self, api_key, client_class, base_url=None):
        super().__init__()

        if not api_key:
            raise ValueError("API key must be provided as an environment variable.")
        if base_url:
            self.client = client_class(api_key=api_key, base_url=base_url)
        else:
            self.client = client_class(api_key=api_key)

        self.model_map = {}

    def get_model_name(self, model):
        return self.model_map.get(model, None)

    def perform_inference(self, model, prompt, timeout, max_output=100, verbosity=True):

        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for provider.")
            start = timer()
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_output,
                # timeout=(10, 100)
            )
            elapsed = timer() - start
            self.log_metrics(model, "response_times", elapsed)
            if verbosity:
                self.display_response(response, elapsed)
            return elapsed
            
        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True, correct_answer=None
    ):
        # print(model, prompt, max_output, verbosity)
        # print("PROMPT")
        # print(prompt)
        score = -1
        generated_text = ""
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for provider.")
            first_token_time = None
            inter_token_latencies = []
            start = timer()
            print(model_id)
            # print("STARTING GROQ REQ")
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                max_tokens=max_output,
                # timeout=(10, 100)
            )
            # print("HERE?")
            print(response)

            for chunk in response:
                # print(chunk)
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
                inter_token_latency = time_to_next_token - prev_token_time
                prev_token_time = time_to_next_token

                inter_token_latencies.append(inter_token_latency)
                token_text = chunk.choices[0].delta.content
                generated_text += token_text
                if verbosity:
                    # print(chunk.choices[0].delta.content or "", end="", flush=True)
                    if len(inter_token_latencies) < 20:
                        print(chunk.choices[0].delta.content or "", end="", flush=True)
                    elif len(inter_token_latencies) == 20:
                        print("...")

            avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            if verbosity:

                print(
                    f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Avg TBT: {avg_tbt:.4f}, Time to First Token (TTFT): {ttft:.4f} seconds, Total Response Time: {elapsed:.4f} seconds"
                )
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timetofirsttoken", ttft)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "response_times", elapsed)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens", inter_token_latencies)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_avg", avg_tbt)
            median = np.percentile(inter_token_latencies, 50)
            p95 = np.percentile(inter_token_latencies, 95)
            p99 = np.percentile(inter_token_latencies, 99)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_median", median)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_p95", p95)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_p99", p99)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "totaltokens", len(inter_token_latencies) + 1)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "tps", (len(inter_token_latencies) + 1) / elapsed)
            self.log_metrics(
                model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "dpsk_output", generated_text
            )
            if correct_answer:
                # print(generated_text, type(generated_text))
                print("-----------------")
                extracted_answer = self.extract_answer_aime(generated_text)

                print(extracted_answer)
                print("-----------------")
                print(correct_answer)
                print("-----------------")
                score = self.calculate_score_aime(extracted_answer, correct_answer)
                print(score)
                print("-----------------")
                if score > -1:
                    self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "accuracy", score)

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return None, None

    def display_response(self, response, elapsed):
        """Display response."""
        print(response.choices[0].message.content)  # [:100] + "...")
        print(f"\nGenerated in {elapsed:.2f} seconds")

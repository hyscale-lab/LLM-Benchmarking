import time
import requests
import numpy as np
from timeit import default_timer as timer
from providers.provider_interface import ProviderInterface
import json

class vLLM(ProviderInterface):
    def __init__(self):
        """
        Initializes the VLLM API server with the necessary configurations.
        """
        super().__init__()

        # Fetch VLLM server configurations from environment variables
        # vllm_host = os.environ.get("vLLM_HOST", "http://10.168.0.28")
        vLLM_PORT = 8000

        self.vllm_port = vLLM_PORT

        # Define available models
        self.model_map = {
            "common-model": "./../scratch/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b",
            "common-model-small": "meta-llama/Llama-3.1-8B-Instruct"
        }

    def get_model_name(self, model):
        """Get the model name, defaulting to 'default-model' if not found."""
        return self.model_map.get(model, "facebook/opt-125m")

    def perform_inference(self, model, prompt, vllm_ip, max_output=100, verbosity=True):
        """
        Sends an inference request to the vLLM API server.
        """
        model_id = self.get_model_name(model)
        formatted_prompt = f"System: {self.system_prompt} \n User: {prompt}"
        print("prompt", formatted_prompt)
        start_time = timer()
        try:
            response = requests.post(
                f"http:/{vllm_ip}:{self.vllm_port}/v1/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": model_id,
                    "prompt": formatted_prompt,
                    "max_tokens": max_output,
                },
                timeout=1800,
            )
            elapsed = timer() - start_time

            # Log response times metric
            self.log_metrics(model, "response_times", elapsed)

            if verbosity:
                print(f"#### _Generated in *{elapsed:.2f}* seconds_")
            
            print(response)
            inference = response.json()
            print(inference)
            return elapsed

        except Exception as e:
            print(f"Error during inference: {e}")
            return None

    def perform_inference_streaming(
        self, model, prompt, vllm_ip, max_output=100, verbosity=True
    ):
        """
        Sends a streaming inference request to the vLLM API server and parses token streams.
        """
        inter_token_latencies = []
        model_id = self.get_model_name(model)
        formatted_prompt = f"System: {self.system_prompt} \n User: {prompt}"
        generated_text = ""

        try:
            start_time = time.perf_counter()
            response = requests.post(
                f"http://{vllm_ip}:{self.vllm_port}/v1/completions",
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "stream": True,
                    "model": model_id,
                    # "messages": [{"role": "user", "content": prompt}],
                    "prompt": formatted_prompt,
                    "max_tokens": max_output,
                },
                stream=True,
                timeout=100,
            )

            first_token_time = None
            for line in response.iter_lines():
                if line:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft = first_token_time - start_time
                        prev_token_time = first_token_time
                        if verbosity:
                            print(line)
                            print(f"##### Time to First Token (TTFT): {ttft:.4f} seconds\n")

                    line_str = line.decode("utf-8").strip()

                    # Handle end of stream
                    if line_str == "data: [DONE]":
                        end_time = time.perf_counter()
                        total_time = end_time - start_time
                        if verbosity:
                            print(f"##### Total Response Time: {total_time:.4f} seconds")
                        break

                    # Extract text from chunk
                    if line_str.startswith("data:"):
                        time_to_next_token = time.perf_counter()
                        inter_token_latency = time_to_next_token - prev_token_time
                        prev_token_time = time_to_next_token
                         
                        chunk = line_str[6:]  # Remove "data: " prefix
                        chunk_json = json.loads(chunk)
                        token_text = chunk_json["choices"][0]["text"]
                        print(token_text, end="")
                        generated_text += token_text  # Accumulate the response
                        # print(token_text, inter_token_latency)
                        inter_token_latencies.append(inter_token_latency)

            avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            if verbosity:
    
                print(
                    f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, "
                    f"Time to First Token (TTFT): {ttft:.4f} seconds, Avg TBT: {avg_tbt}, Total Response Time: {total_time:.4f} seconds"
                )
                print(f"\nGenerated Text: {generated_text}")

            # Log metrics
            self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            median = np.percentile(inter_token_latencies, 50)
            p95 = np.percentile(inter_token_latencies, 95)
            self.log_metrics(model, "timebetweentokens_median", median)
            self.log_metrics(model, "timebetweentokens_p95", p95)
            self.log_metrics(model, "totaltokens", len(inter_token_latencies) + 1)
            self.log_metrics(model, "tps", (len(inter_token_latencies) + 1) / total_time)

            return generated_text, total_time

        except Exception as e:
            print(f"Error during streaming inference: {e}")
            return None, None

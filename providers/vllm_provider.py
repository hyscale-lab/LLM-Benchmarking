import time
import requests
import numpy as np
import asyncio
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
            "common-model-small": "meta-llama/Llama-3.1-8B"
        }

    def initialize_client(self):
        """
        Empty
        """
        pass

    def get_model_name(self, model):
        """Get the model name, defaulting to 'default-model' if not found."""
        return self.model_map.get(model, "facebook/opt-125m")
    
    def normalize_messages(self, messages):
        """
        Empty
        """
        pass
    
    def construct_text_response(self, raw_response):
        """
        Empty
        """
        pass

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

            data = response.json()
            usage = data.get("usage") or {}
            total_tokens = usage.get("completion_tokens")

            tbt = elapsed / max(total_tokens, 1)
            tps = (total_tokens / elapsed)

            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)
            self.log_metrics(model, "response_times", elapsed)

            if verbosity:
                print(f"Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                print(f"#### _Generated in *{elapsed:.2f}* seconds_")

            print(response)
            inference = response.json()
            print(inference)
            return inference

        except Exception as e:
            print(f"Error during inference: {e}")
            return e

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
            response_list = []
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
                        response_list.append(line_str[6:])
                        end_time = time.perf_counter()
                        total_time = end_time - start_time
                        if verbosity:
                            print(f"##### Total Response Time: {total_time:.4f} seconds")
                        break
                    else:
                        response_list.append(json.loads(line_str[6:]))

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

            return response_list

        except Exception as e:
            print(f"Error during streaming inference: {e}")
            return e

    def perform_trace(self, model, proxy_server, load_generator, streaming, num_requests, verbosity, vllm_ip):
        # Set handler for proxy
        async def data_handler(data):
            prompt = data.pop('prompt')
            gen_tokens = data.pop('generated_tokens')

            def inference_sync():
                try:
                    if streaming:
                        response_list = self.perform_inference_streaming(model, prompt, vllm_ip, gen_tokens, verbosity)
                        if isinstance(response_list, Exception):
                            return [{"error": f"Inference failed: {response_list}"}]
                        return response_list
                    else:
                        response = self.perform_inference(model, prompt, vllm_ip, gen_tokens, verbosity)
                        if isinstance(response, Exception):
                            return {"error": f"Inference failed: {response}"}
                        return response

                except Exception as e:
                    print(f"\nData handling failed: {e}")
                    return [{"error": f"Data handling failed: {e}"}] if streaming else {"error": f"Data handling failed: {e}"}

            response = await asyncio.to_thread(inference_sync)
            return response

        proxy_server.set_handler(data_handler)
        proxy_server.set_streaming(streaming)

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

    def perform_multiturn(self, model, time_interval, streaming, num_requests, verbosity, vllm_ip):
        """
        Perform using multiturn input
        """
        def _load_conversation_iterator(path):
            """
            Generator that yields one conversation at a time.
            It does not load the whole file into memory.
            """
            if path is None:
                print("Error: Multiturn dataset path is None.")
                return

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            except FileNotFoundError:
                print(f"Error: Dataset not found at {path}")
                return

        conv_iter = _load_conversation_iterator(self.multiturn_dataset_path)

        for i, conversation in enumerate(conv_iter):
            if i >= num_requests:
                break

            print(f"============ Conversation: {i + 1}/{num_requests} ============")

            current_messages = []

            idx = 0
            while idx < len(conversation):
                # Safety check for pairs
                if idx + 1 >= len(conversation):
                    break

                turn_input = conversation[idx]
                turn_target = conversation[idx + 1]

                if turn_input['role'] == 'human':
                    current_messages.append({
                        "role": "user",
                        "content": turn_input['value']
                    })
                target_tokens = turn_target['generated_tokens']

                # Perform Inference
                print(f"------------ Turn: {(idx // 2) + 1}/{len(conversation) // 2} ------------")
                if streaming:
                    response = self.perform_inference_streaming(
                        model,
                        current_messages,
                        vllm_ip,
                        target_tokens,
                        verbosity
                    )
                else:
                    response = self.perform_inference(
                        model,
                        current_messages,
                        vllm_ip,
                        target_tokens,
                        verbosity
                    )

                if isinstance(response, Exception):
                    print(f"\nTurn failed: {response}")
                    print("Skipping conversation...\n")
                    break

                # Update Context with actual response
                current_messages.append({
                    "role": "assistant",
                    "content": self.construct_text_response(response)
                })

                # Move to next pair
                idx += 2

                # Mimic when human pauses to read response
                time.sleep(time_interval)

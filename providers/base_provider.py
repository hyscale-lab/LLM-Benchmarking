# base_provider.py for chat completions api
from timeit import default_timer as timer
import asyncio
import json
from providers.provider_interface import ProviderInterface


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
        self.timeout = (1, 2)

    def get_model_name(self, model):
        return self.model_map.get(model, None)

    def perform_inference(self, model, prompt, max_output=100, verbosity=True):

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
                timeout=self.timeout
            )
            elapsed = timer() - start

            usage = getattr(response, "usage", None)
            total_tokens = 0
            if usage:
                total_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0

            tbt = elapsed / max(total_tokens, 1)
            tps = total_tokens / elapsed

            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)
            self.log_metrics(model, "response_times", elapsed)

            if verbosity:
                print(f"Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                self.display_response(response, elapsed)
            return elapsed

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for provider.")

            ttft = None
            first_token_time = None
            inter_token_latencies = []
            elapsed = 0.0

            start = timer()
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                max_tokens=max_output,
                timeout=(1, 2)
            )

            for chunk in response:
                if timer() - start > 90:
                    elapsed = timer() - start
                    print("[WARN] Streaming exceeded 90s, stopping early.")
                    break

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
                if verbosity:
                    # print(chunk.choices[0].delta.content or "", end="", flush=True)
                    if len(inter_token_latencies) < 20:
                        print(chunk.choices[0].delta.content or "", end="", flush=True)
                    elif len(inter_token_latencies) == 20:
                        print("...")

            token_count = (len(inter_token_latencies) + 1) if ttft is not None else 0
            non_first_latency = max(elapsed - (ttft or 0.0), 0.0)
            avg_tbt = (non_first_latency / (token_count)) if token_count > 0 else 0.0
            
            if verbosity:

                print(
                    f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Avg TBT: {avg_tbt:.4f}, Time to First Token (TTFT): {ttft:.4f} seconds, Total Response Time: {elapsed:.4f} seconds"
                )
            if ttft is not None:
                self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", elapsed)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            self.log_metrics(model, "totaltokens", token_count)
            self.log_metrics(model, "tps", (token_count / elapsed) if elapsed > 0 else 0.0)

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return None, None

    def perform_trace_mode(self, proxy_server, load_generator, num_requests, verbosity):
        # Set handler for proxy
        async def data_handler(data, streaming):
            if streaming:
                print("\nRequest not sent. Streaming not allowed in trace mode.")
                return [{"error": "Streaming not allowed in trace mode."}]

            def inference_sync():
                try:
                    model_id = data.get('model')
                    if not model_id or model_id not in self.model_map.values():
                        raise Exception(f"Model {model_id} not found in model map.")
                    model = next((k for k, v in self.model_map.items() if v == model_id))
                    if 'timeout' not in data:
                        data['timeout'] = (1, 2)

                    # Non-streaming inference
                    start_time = timer()
                    response = self.client.chat.completions.create(**data)
                    elapsed_time = timer() - start_time

                    usage = getattr(response, "usage", None)
                    total_tokens = 0
                    if usage:
                        total_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
                    tbt = elapsed_time / max(total_tokens, 1)
                    tps = (total_tokens / elapsed_time)
                    self.log_metrics(model, "totaltokens", total_tokens)
                    self.log_metrics(model, "timebetweentokens", tbt)
                    self.log_metrics(model, "tps", tps)
                    self.log_metrics(model, "response_times", elapsed_time)

                    if verbosity:
                        print()
                        print(f"##### Generated in {elapsed_time:.2f} seconds")
                        print(f"##### Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                        print(f"Response: {response.choices[0].message.content}")

                    return json.loads(response.json())

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

    def display_response(self, response, elapsed):
        """Display response."""
        print(response.choices[0].message.content)  # [:100] + "...")
        print(f"\nGenerated in {elapsed:.2f} seconds")

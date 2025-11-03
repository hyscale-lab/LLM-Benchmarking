# anthropic_provider.py
import os
import asyncio
from timeit import default_timer as timer
import anthropic
from providers.provider_interface import ProviderInterface


class Anthropic(ProviderInterface):
    def __init__(self):
        """
        Initializes the AnthropicProvider with the necessary API key and client.
        """
        super().__init__()

        # Load API key from environment
        self.api_key = os.getenv("ANTHROPIC_API")
        if not self.api_key:
            raise ValueError("API key must be provided as an environment variable.")

        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Model mapping for Anthropic models
        self.model_map = {

            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",  # approx 70b
            "claude-3-opus": "claude-3-opus-20240229",  # approx 2T
            "claude-3-haiku": "claude-3-5-haiku-20241022",  # approx 20b
            "common-model": "claude-3-5-sonnet-20241022",
        }

    def get_model_name(self, model):
        """
        Retrieves the actual model identifier for a given model alias.

        Args:
            model (str): The alias name of the model.

        Returns:
            str: The identifier of the model for API calls.
        """
        return self.model_map.get(model, None)

    def perform_inference(self, model, prompt, max_output=100, verbosity=True):
        """
        Performs a synchronous inference call to the Anthropic API.

        Args:
            model (str): The model name to use for inference.
            prompt (str): The user prompt for the chat completion.

        Returns:
            float: The elapsed time in seconds for the inference request.
        """
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for Anthropic.")

            start = timer()
            response = self.client.messages.create(
                model=model_id,
                max_tokens=max_output,
                messages=[{"role": "user", "content": prompt}],
                # temperature=0.7,
                stop_sequences=["\nUser:"],
                timeout=500,
            )
            elapsed = timer() - start

            usage = getattr(response, "usage", None)
            total_tokens = (getattr(usage, "output_tokens", 0) or 0) if usage else 0

            tbt = elapsed / max(total_tokens, 1)
            tps = (total_tokens / elapsed)

            self.log_metrics(model, "response_times", elapsed)
            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)
            # Process and display the response
            if verbosity:
                print(f"Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                self.display_response(response, elapsed)
            return elapsed

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None, None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        """
        Performs a streaming inference call to the Anthropic API.

        Args:
            model (str): The model name to use for inference.
            prompt (str): The user prompt for the chat completion.
        """
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for Anthropic.")

            first_token_time = None
            inter_token_latencies = []
            TTFT = None

            start = timer()
            with self.client.messages.stream(
                model=model_id,
                max_tokens=max_output,
                messages=[{"role": "user", "content": prompt}],
                # temperature=0.7,
                stop_sequences=["\nUser:"],
                timeout=500,
            ) as stream:
                for chunk in stream.text_stream:
                    if timer() - start > 90:
                        elapsed = timer() - start
                        print("[WARN] Streaming exceeded 90s, stopping early.")
                        break
                    if first_token_time is None:
                        first_token_time = timer()
                        TTFT = first_token_time - start
                        prev_token_time = first_token_time
                        self.log_metrics(model, "timetofirsttoken", TTFT)
                        if verbosity:
                            print(f"\nTime to First Token (TTFT): {TTFT:.4f} seconds\n")

                    # Calculate inter-token latencies
                    time_to_next_token = timer()
                    inter_token_latency = time_to_next_token - prev_token_time
                    prev_token_time = time_to_next_token

                    inter_token_latencies.append(inter_token_latency)
                    if verbosity:
                        if len(inter_token_latencies) < 20:
                            print(chunk, end="", flush=True)
                        elif len(inter_token_latencies) == 20:
                            print("...")

                elapsed = timer() - start
                if verbosity:
                    print(f"\nTotal Response Time: {elapsed:.4f} seconds")
                    print(f"Total tokens: {len(inter_token_latencies)}")

            # Log remaining metrics
            token_count = len(inter_token_latencies) + (1 if TTFT is not None else 0)
            if TTFT is None:
                avg_tbt = 0.0
            else:
                non_first_latency = max(elapsed - TTFT, 0.0)
                avg_tbt = non_first_latency / max(token_count - 1, 1)

            if verbosity:
                print(f"Avg TBT: {avg_tbt:.4f} seconds")

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
                    data.pop('stream')
                    model_id = data.get('model')
                    if not model_id or model_id not in self.model_map.values():
                        raise Exception(f"Model {model_id} not found in model map.")
                    model = next((k for k, v in self.model_map.items() if v == model_id))
                    if 'timeout' not in data:
                        data['timeout'] = 500

                    # Non-streaming inference
                    start_time = timer()
                    response = self.client.messages.create(**data)
                    elapsed_time = timer() - start_time

                    usage = getattr(response, "usage", None)
                    total_tokens = (getattr(usage, "output_tokens", 0) or 0) if usage else 0
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
                        print("Response: ", end="")
                        for block in response.content:
                            print(block.text)

                    return response.model_dump()

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
        """
        Prints the response content and the time taken to generate it.

        Args:
            response (dict): The response dictionary from the Anthropic API.
            elapsed (float): Time in seconds taken to generate the response.
        """
        # content = "".join(block.get("text", "") for block in response.content)
        # print(response)
        for block in response.content:
            print(block.text)
        print(f"\nGenerated in {elapsed:.2f} seconds")

# anthropic_provider.py
import os
import anthropic
import numpy as np
from timeit import default_timer as timer
from providers.provider_interface import ProviderInterface
import math

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
            "claude-3-haiku": "claude-3-5-haiku-latest",  # approx 20b
            "common-model": "claude-3-5-sonnet-20241022",
            "common-model-small": "claude-3-5-haiku-latest"
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
            self.log_metrics(model, "response_times", elapsed)
            # Process and display the response
            if verbosity:
                self.display_response(response, elapsed)
            return elapsed

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None, None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True, correct_answer=None
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
            c = 0
            start = timer()
            print("ENTERING")
            with self.client.messages.stream(
                model=model_id,
                max_tokens=max_output,
                messages=[{"role": "user", "content": prompt}],
                # temperature=0.7,
                stop_sequences=["\nUser:"],
                timeout=500,
            ) as stream:
                # for chunk in stream.text_stream:
                # for event in stream:
                #     # maybe check for a stop event
                #     if event == "MessageStopEvent":
                #         print(event)
                #         print("end")
                #         break

                for chunk in stream.text_stream:
                    # print(chunk)
                    
                    if first_token_time is None:
                        first_token_time = timer()
                        ttft = first_token_time - start
                        prev_token_time = first_token_time
                        self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timetofirsttoken", ttft)
                        if verbosity:
                            print(f"\nTime to First Token (TTFT): {ttft:.4f} seconds\n")
            
                    # Calculate inter-token latencies
                    time_to_next_token = timer()
                    # inter_token_latency = time_to_next_token - prev_token_time
                    elapsed = time_to_next_token - prev_token_time
                    token_count = len(chunk.split())
                    c += token_count
                    avg_latency = elapsed / max(1, token_count)
                    # print(f"Len: {token_count}, {chunk}", end="~")

                    # record one latency entry *per* token
                    inter_token_latencies.extend([avg_latency] * token_count)
                    prev_token_time = time_to_next_token
                    print(chunk, end="", flush=True)
                    # inter_token_latencies.append(inter_token_latency)
                    # if verbosity:
                    #     if len(inter_token_latencies) < 20:
                    #         print(chunk, end="", flush=True)
                    #     elif len(inter_token_latencies) == 20:
                    #         print("...")

                elapsed = timer() - start
                if verbosity:
                    print(f"\nTotal Response Time: {elapsed:.4f} seconds")
                    print(f"Total tokens: {len(inter_token_latencies), c}")
                    # print(
                    #     f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Avg TBT: {avg_tbt:.4f}, Time to First Token (TTFT): {ttft:.4f} seconds, Total Response Time: {elapsed:.4f} seconds"
                    # )

            # Log remaining metrics
            # avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            # avg_tbt = sum(inter_token_latencies) / max_output

            # self.log_metrics(model, "response_times", elapsed)
            # self.log_metrics(model, "timebetweentokens", avg_tbt)
            # self.log_metrics(model, "totaltokens", len(inter_token_latencies) + 1)
            # self.log_metrics(model, "tps", (len(inter_token_latencies) + 1) / elapsed)
            # self.log_metrics(
            #     model, "timebetweentokens_median", np.percentile(inter_token_latencies, 50)
            # )
            # self.log_metrics(
            #     model, "timebetweentokens_p95", np.percentile(inter_token_latencies, 95)
            # )

            avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            print("HI!!!!")
            if verbosity:

                print(
                    f"\nNumber of output tokens/chunks: {len(inter_token_latencies) + 1}, Avg TBT: {avg_tbt:.4f}, Total Response Time: {elapsed:.4f} seconds"
                )
            
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



        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return None, None
            
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

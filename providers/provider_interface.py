import os
import time
import json
import asyncio
import threading
from abc import ABC, abstractmethod


# create an interface for providers (abstract class)
class ProviderInterface(ABC):

    def __init__(self):
        """
        Initializes the Provider with the necessary API key and client.
        """
        # experiment constants
        self.min_tokens = 10000
        self.system_prompt = (
            f"Please provide a detailed response of MORE THAN {self.min_tokens} words"
        )

        # metrics
        self.metrics = {
            "response_times": {},
            "response_times_median": {},
            "response_times_p95": {},
            "timetofirsttoken": {},
            "timetofirsttoken_median": {},
            "timetofirsttoken_p95": {},
            "totaltokens": {},
            "tps": {},
            "tps_median": {},
            "tps_p95": {},
            "timebetweentokens": {},
            "timebetweentokens_median": {},
            "timebetweentokens_p95": {},
            "aime_2024_accuracy": {}
        }

        # for trace input type
        self._lock = threading.Lock()
        self.trace_dataset_path = os.getenv('TRACE_DATASET_PATH')
        self.trace_result_path = f'./trace/{self.__class__.__name__}.result'

        # for multiturn input type
        self.multiturn_dataset_path = os.getenv('MULTITURN_DATASET_PATH')

    def log_metrics(self, model_name, metric, value):
        """
        Logs metrics
        """
        with self._lock:
            if metric not in self.metrics:
                raise ValueError(f"Metric type '{metric}' is not defined.")
            if model_name not in self.metrics[metric]:
                self.metrics[metric][model_name] = []

            self.metrics[metric][model_name].append(value)

    @abstractmethod
    def initialize_client(self):
        """
        initialize client
        """

    @abstractmethod
    def get_model_name(self, model):
        """
        get model names
        """

    @abstractmethod
    def perform_inference(self, model, messages, max_output, verbosity):
        """
        perform_inference
        """

    @abstractmethod
    def perform_inference_streaming(self, model, messages, max_output, verbosity):
        """
        perform_inference_streaming
        """

    def perform_trace(self, model, proxy_server, load_generator, streaming, num_requests, verbosity):
        """
        Perform using trace input
        """
        # Set handler for proxy
        async def data_handler(data):
            prompt = data.pop('prompt')
            gen_tokens = data.pop('generated_tokens')

            def inference_sync():
                try:
                    if streaming:
                        response_list = self.perform_inference_streaming(model, prompt, gen_tokens, verbosity)
                        if isinstance(response_list, Exception):
                            return [{"error": f"Inference failed: {response_list}"}]
                        return response_list
                    else:
                        response = self.perform_inference(model, prompt, gen_tokens, verbosity)
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

        # For plots from load generator
        os.makedirs('plots', exist_ok=True)

        # Start load generator
        load_generator.send_loads(
            self.trace_dataset_path,
            self.trace_result_path,
            sampling_rate=200,
            recur_step=3,
            limit=num_requests,
            max_drift=1000,
        )

    @abstractmethod
    def normalize_messages(self, messages):
        """
        Standardizes varied input formats into a list of compatible message dictionaries.

        This function accepts either a single prompt string or a list of message dictionaries.
        If a list is provided, it normalizes role names (e.g., converting 'assistant' to 'model')
        to ensure compatibility with LLM provider APIs.
        """

    @abstractmethod
    def construct_text_response(self, raw_response):
        """
        Construct text response from raw response
        """

    def perform_multiturn(self, model, time_interval, streaming, num_requests, verbosity):
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

        request_id = 0
        max_retries = 3
        for i, conversation in enumerate(conv_iter):
            print(f"============ Conversation: {i + 1} ============")

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
                else:
                    print(f"Unexpected role: {turn_input['role']}")

                target_tokens = turn_target['generated_tokens']

                # Perform Inference
                print(f"------------ Turn: {(idx // 2) + 1}/{len(conversation) // 2} ------------")
                for attempt in range(max_retries):
                    if attempt > 0:
                        print(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(5)

                    if streaming:
                        response = self.perform_inference_streaming(
                            model,
                            current_messages,
                            target_tokens,
                            verbosity
                        )
                    else:
                        response = self.perform_inference(
                            model,
                            current_messages,
                            target_tokens,
                            verbosity
                        )

                    if isinstance(response, Exception):
                        print(f"\nAttempt {attempt + 1} failed: {response}")
                        continue

                    break  # Turn success. EXIT for loop, SKIP else to next turn
                else:
                    print("Turn failed. Skipping conversation...\n")
                    break  # Turn failed. SKIP conversation

                # Update Context with actual response
                current_messages.append({
                    "role": "assistant",
                    "content": self.construct_text_response(response)
                })

                request_id += 1

                # Check num requests limit
                if request_id == num_requests:
                    print("\nRequest limit hit. Stopping...\n")
                    return

                # Move to next pair
                idx += 2

                # Mimic when human pauses to read response
                time.sleep(time_interval)

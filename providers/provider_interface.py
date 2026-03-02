import os
import time
import json
import csv
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
            "vision_encoder_timetofirsttoken": {},
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

        # for vqa input type
        self.vqa_dataset_path = os.getenv('VQA_DATASET_PATH')
        self.vqa_log_path = './vqa/logs'

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
            sampling_rate=100,
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
                print(f"Error: Multiturn dataset not found at {path}")
                return

        conv_iter = _load_conversation_iterator(self.multiturn_dataset_path)

        max_retries = 3
        for i, conversation in enumerate(conv_iter):
            if i == num_requests:
                print("\nRequest limit hit. Stopping...\n")
                return

            print(f"============ Request: {i + 1} ============")

            # Construct messages
            messages = []
            idx = 0
            while idx < len(conversation):
                # Safety check for pairs
                if idx + 1 >= len(conversation):
                    print(f"Unexpected length: {len(conversation)}")
                    break

                turn_input = conversation[idx]
                turn_response = conversation[idx + 1]

                if turn_input['role'] == 'human':
                    messages.append({
                        "role": "user",
                        "content": turn_input['value']
                    })
                else:
                    print(f"Unexpected role: {turn_input['role']}")

                if turn_response['role'] == 'gpt':
                    if 'value' in turn_response:
                        messages.append({
                            "role": "assistant",
                            "content": turn_response['value']
                        })
                    elif 'generated_tokens' in turn_response:
                        target_tokens = turn_response['generated_tokens']
                    else:
                        print(f"Missing keys: value and generated_tokens")
                else:
                    print(f"Unexpected role: {turn_input['role']}")

                idx += 2

            # Perform Inference
            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)

                if streaming:
                    response = self.perform_inference_streaming(
                        model,
                        messages,
                        target_tokens,
                        verbosity
                    )
                else:
                    response = self.perform_inference(
                        model,
                        messages,
                        target_tokens,
                        verbosity
                    )

                if isinstance(response, Exception):
                    print(f"\nAttempt {attempt + 1} failed: {response}")
                    continue

                break

            else:
                print("Request failed. Skipping request...\n")

                # Mimic when human pauses to read response
                time.sleep(time_interval)

    def perform_vqa(self, model, streaming, num_requests, verbosity):
        """
        Perform using vqa input
        """
        def _load_vqa_iterator(path):
            """
            Generator that yields one VQA benchmark sample at a time.
            Streams directly from the JSONL file with near-zero memory footprint.
            """
            if path is None:
                print("Error: VQA dataset path is None.")
                return

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            except FileNotFoundError:
                print(f"Error: VQA dataset not found at {path}")
                return

        # Initialize a log file
        safe_model_name = self.get_model_name(model).replace("/", "_").replace(":", "_")
        csv_filename = f"vqa_ttft_results_{self.__class__.__name__}_{safe_model_name}.csv"

        print(f"Initializing log file: {csv_filename}")
        os.makedirs(self.vqa_log_path, exist_ok=True)
        with open(os.path.join(self.vqa_log_path, csv_filename), mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "multimodal_ttft", "text_only_ttft", "vision_encoder_ttft"])

        vqa_iter = _load_vqa_iterator(self.vqa_dataset_path)

        self.system_prompt = "Analyze image. Answer with exactly one word. No sentences. No punctuation."
        max_retries = 3
        for i, item in enumerate(vqa_iter):
            if i >= num_requests:
                print("\nRequest limit hit. Stopping...\n")
                break

            print(f"============ VQA Item: {i + 1}/{num_requests}  ============")

            vqa_dataset_dir = os.path.dirname(self.vqa_dataset_path)
            question_id = item["question_id"]
            image_path = os.path.join(vqa_dataset_dir, item["image_path"])
            question = item["question"]
            dummy_text = item["dummy_text"]

            # Prepare messages
            multimodal_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_path": image_path},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            text_only_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": dummy_text}]
                }
            ]

            # --- PASS 1 ---
            print("--> Running Multimodal Pass...")
            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)

                # We only generate small number of tokens since we focus strictly on TTFT.
                if streaming:
                    response = self.perform_inference_streaming(
                        model,
                        multimodal_messages,
                        max_output=10,
                        verbosity=verbosity
                    )
                else:
                    print("Non-streaming not supported.")
                    return

                if isinstance(response, Exception):
                    print(f"Multimodal attempt {attempt + 1} failed: {response}")
                    continue

                break  # Success, exit retry loop
            else:
                print("Multimodal passes failed. Skipping sample...\n")
                continue

            # --- PASS 2 ---
            print("--> Running Text-Only Baseline Pass...")
            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"Retrying... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)

                if streaming:
                    response = self.perform_inference_streaming(
                        model,
                        text_only_messages,
                        max_output=10,
                        verbosity=verbosity
                    )
                else:
                    print("Non-streaming not supported.")
                    return

                if isinstance(response, Exception):
                    print(f"\nText attempt {attempt + 1} failed: {response}")
                    continue

                break  # Success, exit retry loop
            else:
                print("Text passes failed. Skipping sample...\n")
                continue

            # --- CALCULATE VISION ENCODER LATENCY ---
            ttft_multimodal = self.metrics['timetofirsttoken'][model][-2]
            ttft_text = self.metrics['timetofirsttoken'][model][-1]

            ttft_vision_encoder = ttft_multimodal - ttft_text

            if verbosity:
                print(f"Results for Item {i + 1}:")
                print(f"  Total Multimodal TTFT : {ttft_multimodal:.4f} s")
                print(f"  Text-Only Prefill TTFT: {ttft_text:.4f} s")
                print(f"  Isolated Vision TTFT  : {ttft_vision_encoder:.4f} s")

            self.log_metrics(model, "vision_encoder_timetofirsttoken", ttft_vision_encoder)

            # Append results to log
            with open(os.path.join(self.vqa_log_path, csv_filename), mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{question_id}",
                    f"{ttft_multimodal}",
                    f"{ttft_text}",
                    f"{ttft_vision_encoder}"
                ])

            # Optional cooldown to let VRAM flush and prevent thermal throttling
            time.sleep(15)

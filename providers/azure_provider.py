import os
import asyncio
from time import perf_counter as timer
import numpy as np
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from providers.base_provider import ProviderInterface


class Azure(ProviderInterface):
    def __init__(self):
        """Initialize AzureProvider with required API information."""
        super().__init__()

        self.endpoint = os.getenv("AZURE_AI_ENDPOINT")
        self.api_key = os.getenv("AZURE_AI_API_KEY")

        # Map model names to Azure model IDs
        self.model_map = {
            # "mistral-7b-instruct-v0.1": "mistral-7b-instruct-v0.1",
            "meta-llama-3.1-8b-instruct": "Meta-Llama-3.1-8B-Instruct-fyp",
            "meta-llama-3.1-70b-instruct": "Meta-Llama-3-1-70B-Instruct-fyp",
            "mistral-large": "Mistral-Large-2411-yatcd",
            "common-model": "Mistral-Large-2411-yatcd",
        }

        self._client = None

    def _ensure_client(self):
        """
        Create the Azure client only when first used.
        Raise a clear error if env vars are missing.
        """
        if self._client is not None:
            return

        if not self.api_key or not isinstance(self.api_key, str):
            raise RuntimeError(
                "Azure provider misconfigured: AZURE_AI_API_KEY is missing or not a string."
            )
        if not self.endpoint:
            raise RuntimeError(
                "Azure provider misconfigured: AZURE_AI_ENDPOINT is missing."
            )

        credential = AzureKeyCredential(self.api_key)
        self._client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=credential,
            api_version="2024-05-01-preview",
        )

    def get_model_name(self, model):
        """Retrieve the model name based on the input key."""
        return self.model_map.get(model, None)

    def perform_inference(self, model, prompt, max_output=100, verbosity=True):
        """Performs non-streaming inference request to Azure."""
        try:
            self._ensure_client()
            client = self._client
            model_id = self.get_model_name(model)
            if model_id is None:
                print(f"Model {model} not available.")
                return None
            start_time = timer()
            response = client.complete(
                messages=[
                    SystemMessage(content=self.system_prompt),
                    UserMessage(content=prompt),
                ],
                max_tokens=max_output,
                model=model_id
            )
            elapsed = timer() - start_time

            usage = response.get("usage")
            total_tokens = usage.get("completion_tokens") or 0
            tbt = elapsed / max(total_tokens, 1)
            tps = (total_tokens / elapsed)

            self.log_metrics(model, "response_times", elapsed)
            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)

            if verbosity:
                print(f"Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                print(f"Response: {response['choices'][0]['message']['content']}")
            return response

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return None, None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        """Performs streaming inference request to Azure."""
        self._ensure_client()
        client = self._client
        model_id = self.get_model_name(model)
        if model_id is None:
            print(f"Model {model} not available.")
            return None

        inter_token_latencies = []
        start_time = timer()
        try:
            first_token_time = None
            with client.complete(
                stream=True,
                messages=[
                    SystemMessage(content=self.system_prompt),
                    UserMessage(content=prompt),
                ],
                max_tokens=max_output,
                model=model_id
            ) as response:
                for event in response:
                    if not event.choices or not event.choices[0].delta:
                        continue

                    delta = event.choices[0].delta
                    if delta.content:
                        if first_token_time is None:
                            first_token_time = timer()
                            ttft = first_token_time - start_time
                            prev_token_time = first_token_time
                            print(f"##### Time to First Token (TTFT): {ttft:.4f} seconds\n")

                        time_to_next_token = timer()
                        inter_token_latency = time_to_next_token - prev_token_time
                        prev_token_time = time_to_next_token
                        inter_token_latencies.append(inter_token_latency)

                        print(delta.content, end="", flush=True)

            total_time = timer() - start_time
            # Calculate total metrics

            if verbosity:
                print(f"\nTotal Response Time: {total_time:.4f} seconds")
                print(len(inter_token_latencies))

            # Log metrics
            avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
            print(f"{avg_tbt:.4f}, {len(inter_token_latencies)}")
            self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            self.log_metrics(
                model, "timebetweentokens_median", np.median(inter_token_latencies)
            )
            self.log_metrics(
                model, "timebetweentokens_p95", np.percentile(inter_token_latencies, 95)
            )
            self.log_metrics(model, "totaltokens", len(inter_token_latencies) + 1)

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

                    # Format prompt messages
                    for i, m in enumerate(data['messages']):
                        match m['role']:
                            case 'system':
                                data['messages'][i] = SystemMessage(m['content'])
                            case 'assistant':
                                data['messages'][i] = AssistantMessage(m['content'])
                            case 'user':
                                data['messages'][i] = UserMessage(m['content'])
                            case _:
                                raise Exception(f"Role {m['role']} not supported.")

                    # Non-streaming inference
                    self._ensure_client()
                    start_time = timer()
                    response = self._client.complete(**data)
                    elapsed_time = timer() - start_time

                    usage = response.get("usage")
                    total_tokens = usage.get("completion_tokens") or 0
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
                        print(f"Response: {response}")

                    return response.to_dict()

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

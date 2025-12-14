import os
import asyncio
from utils.accuracy_mixin import AccuracyMixin
from time import perf_counter as timer
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from providers.base_provider import ProviderInterface
from openai import AzureOpenAI


class Azure(AccuracyMixin, ProviderInterface):
    def __init__(self):
        """Initialize AzureProvider with required API information."""
        super().__init__()

        self.endpoint = os.getenv("AZURE_AI_ENDPOINT")
        self.api_key = os.getenv("AZURE_AI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        # Map model names to Azure model IDs
        self.model_map = {
            "llama-3.3-70b-instruct": "Llama-3.3-70B-Instruct",
            "meta-llama-3.1-8b-instruct": "Meta-Llama-3.1-8B-Instruct",
            "common-model": "Meta-Llama-3.1-8B-Instruct",
            "reasoning-model": ["o4-mini", "gpt-4o"],
        }

        self._client = None
        self._openai_client = None

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
        self._openai_client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.openai_endpoint,
            api_key=self.api_key,
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
            return response.as_dict()

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return e

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
        first_token_time = None
        ttft = None
        try:
            with client.complete(
                stream=True,
                messages=[
                    SystemMessage(content=self.system_prompt),
                    UserMessage(content=prompt),
                ],
                max_tokens=max_output,
                model=model_id
            ) as response:
                response_list = []
                for event in response:
                    if timer() - start_time > 90:
                        print("[WARN] Streaming exceeded 90s, stopping early.")
                        break

                    response_list.append(event.as_dict())
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
            token_count = (len(inter_token_latencies) + 1) if ttft is not None else 0
            non_first_latency = max(total_time - (ttft or 0.0), 0.0)
            avg_tbt = (non_first_latency / (token_count)) if token_count > 0 else 0.0

            if verbosity:
                print(f"\nTotal Response Time: {total_time:.4f} seconds")
                print(f"Total tokens: {token_count}")
                print(f"Avg TBT: {avg_tbt:.4f} seconds")

            # Log metrics
            if ttft is not None:
                self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            self.log_metrics(model, "totaltokens", token_count)

            return response_list

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return e

    def perform_trace_mode(self, proxy_server, load_generator, num_requests, streaming, verbosity, model='common-model'):
        # Set handler for proxy
        async def data_handler(data):
            gen_tokens = data.pop('generated_tokens')
            prompt = data.pop('prompt')

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

    def _to_azure_messages(self, messages):
        out = []
        for m in messages:
            role = (m.get("role") or "").lower()
            content = m.get("content", "")
            if role == "system":
                out.append(SystemMessage(content=content))
            else:
                out.append(UserMessage(content=content))
        return out

    def _chat_for_eval(self, model_id, messages):
        self._ensure_client()
        client = self._openai_client
        azure_msgs = self._to_azure_messages(messages)
        max_tokens = 40000 if model_id == "o4-mini" else 16384
        start = timer()
        try:
            resp = client.chat.completions.create(
                messages=azure_msgs,
                max_completion_tokens=max_tokens,
                model=model_id
            )
            text = resp.choices[0].message.content
            elapsed = timer() - start

            tokens = 0
            try:
                usage = getattr(resp, "usage", None) or {}
                tokens = (
                    getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", None)
                    or (isinstance(usage, dict) and (usage.get("completion_tokens") or usage.get("output_tokens")))
                    or 0
                )
            except Exception:
                tokens = 0

            return text, int(tokens or 0), float(elapsed)

        except Exception as e:
            elapsed = timer() - start
            print(f"ERROR IS {e!r}")
            return "", 0, float(elapsed)

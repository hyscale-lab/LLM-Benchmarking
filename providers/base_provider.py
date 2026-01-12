# base_provider.py for chat completions api
from timeit import default_timer as timer
import json
from providers.provider_interface import ProviderInterface
from utils.accuracy_mixin import AccuracyMixin


class BaseProvider(AccuracyMixin, ProviderInterface):
    def __init__(self, api_key, client_class, base_url=None):
        super().__init__()

        if not api_key:
            raise ValueError("API key must be provided as an environment variable.")
        if base_url:
            self.client = client_class(api_key=api_key, base_url=base_url, max_retries=0)
        else:
            self.client = client_class(api_key=api_key, max_retries=0)

        self.model_map = {}
        self.timeout = 500

    def get_model_name(self, model):
        return self.model_map.get(model, None)

    def normalize_messages(self, messages):
        if isinstance(messages, str):
            normalized_msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": messages}
            ]
        elif isinstance(messages, list):
            normalized_msgs = [{"role": "system", "content": self.system_prompt}]
            for msg in messages:
                role = msg["role"]

                if role in ["user", "assistant"]:
                    normalized_msgs.append(msg)
                else:
                    print(f"Invalid role found in messages: {role}")

        return normalized_msgs
    
    def construct_text_response(self, raw_response):
        if isinstance(raw_response, dict):
            text_response = raw_response['choices'][0]['message']['content']
        elif isinstance(raw_response, list):
            text_response = "".join(
                block['choices'][0]['delta']['content']
                for block in raw_response
                if block['choices'][0]['delta']['content']
            )

        return text_response

    def perform_inference(self, model, messages, max_output=100, verbosity=True):

        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for provider.")
            start = timer()
            response = self.client.chat.completions.create(
                model=model_id,
                messages=self.normalize_messages(messages),
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
            return json.loads(response.json())

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return e

    def perform_inference_streaming(
        self, model, messages, max_output=100, verbosity=True
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
                messages=self.normalize_messages(messages),
                stream=True,
                max_tokens=max_output,
                timeout=self.timeout
            )

            response_list = []
            for chunk in response:
                response_list.append(json.loads(chunk.json()))
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
            return response_list

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return e

    def display_response(self, response, elapsed):
        """Display response."""
        print(response.choices[0].message.content)  # [:100] + "...")
        print(f"\nGenerated in {elapsed:.2f} seconds")

    def _chat_for_eval(self, model_id, messages):
        start = timer()
        try:
            resp = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=40960,
                temperature=0,
            )
            elapsed = timer() - start

            text = ""
            try:
                choice = resp.choices[0]
                msg = getattr(choice, "message", None)
                text = (getattr(msg, "content", "") or getattr(choice, "text", "") or "")
            except Exception:
                text = ""

            tokens = 0
            try:
                usage = getattr(resp, "usage", None)
                tokens = (
                    getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", None)
                    or 0
                )
            except Exception:
                tokens = 0

            return text, int(tokens or 0), float(elapsed)

        except Exception as e:
            elapsed = timer() - start
            print(f"[ERROR] _chat_for_eval failed (BaseProvider): {e}")
            return "", 0, float(elapsed)

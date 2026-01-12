# anthropic_provider.py
import os
from timeit import default_timer as timer
import anthropic
from providers.provider_interface import ProviderInterface
from utils.accuracy_mixin import AccuracyMixin


class Anthropic(AccuracyMixin, ProviderInterface):
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
            "claude-3-opus": "claude-3-opus-20240229",  # approx 2T
            "claude-3-haiku": "claude-3-5-haiku-20241022",  # approx 20b
            "common-model": "claude-3-5-haiku-20241022",
            "reasoning-model": ["claude-sonnet-4-5-20250929"]
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

    def normalize_messages(self, messages):
        if isinstance(messages, str):
            normalized_msgs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            normalized_msgs = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role in ["user", "assistant"]:
                    normalized_msgs.append({"role": role, "content": content})
                else:
                    print(f"Invalid role found in messages: {role}")

        return normalized_msgs
    
    def construct_text_response(self, raw_response):
        if isinstance(raw_response, dict):
            text_response = ""
            for block in raw_response['content']:
                text_response += block['text']
        elif isinstance(raw_response, list):
            text_response = "".join(raw_response)

        return text_response

    def perform_inference(self, model, messages, max_output=100, verbosity=True):
        """
        Performs a synchronous inference call to the Anthropic API.

        Args:
            model (str): The model name to use for inference.
            messages: The messages for the chat completion.
        """
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for Anthropic.")

            start = timer()
            response = self.client.messages.create(
                system=self.system_prompt,
                model=model_id,
                max_tokens=max_output,
                messages=self.normalize_messages(messages),
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
            return response.model_dump()

        except Exception as e:
            print(f"[ERROR] Inference failed for model '{model}': {e}")
            return e

    def perform_inference_streaming(
        self, model, messages, max_output=100, verbosity=True
    ):
        """
        Performs a streaming inference call to the Anthropic API.

        Args:
            model (str): The model name to use for inference.
            messages: The messages for the chat completion.
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
                system=self.system_prompt,
                model=model_id,
                max_tokens=max_output,
                messages=self.normalize_messages(messages),
                # temperature=0.7,
                stop_sequences=["\nUser:"],
                timeout=500,
            ) as stream:
                response_list = []
                for chunk in stream.text_stream:
                    response_list.append(chunk)
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
            return response_list

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return e

    @staticmethod
    def _split_system_and_messages(messages):
        system_text = None
        out = []
        for m in messages:
            role = (m.get("role") or "").lower()
            content = m.get("content", "")
            if role == "system":
                system_text = f"{system_text}\n{content}".strip() if system_text else content
            else:
                r = role if role in ("user", "assistant") else "user"
                out.append({"role": r, "content": content})
        return system_text, out

    def _chat_for_eval(self, model_id, messages):
        system_text, msg_list = self._split_system_and_messages(messages)

        start = timer()
        try:
            resp = self.client.messages.create(
                model=model_id,
                system=system_text,
                messages=msg_list,
                max_tokens=4096,
                temperature=0,
            )
            elapsed = timer() - start

            raw = ""
            for b in (resp.content or []):
                if getattr(b, "type", None) == "text":
                    raw += b.text

            tokens = 0
            try:
                usage = getattr(resp, "usage", None)
                tokens = getattr(usage, "output_tokens", 0) if usage else 0
            except Exception:
                tokens = 0

            return raw, int(tokens or 0), float(elapsed)

        except Exception as e:
            elapsed = timer() - start
            print(f"[ERROR] _chat_for_eval failed: {e}")
            return "", 0, float(elapsed)

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

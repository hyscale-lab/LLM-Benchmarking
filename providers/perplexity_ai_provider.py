import os
import json
from timeit import default_timer as timer
from openai import OpenAI
from providers.base_provider import BaseProvider


class PerplexityAI(BaseProvider):
    """perplexity provider class"""

    def __init__(self):
        """
        Initializes the AnthropicProvider with the necessary API key and client.
        """
        perplexity_api = os.environ.get("PERPLEXITY_AI_API")

        if not perplexity_api:
            raise ValueError(
                "Perplexity AI API token must be provided as an environment variable."
            )

        client_class = OpenAI
        base_url = "https://api.perplexity.ai"

        super().__init__(
            api_key=perplexity_api, client_class=client_class, base_url=base_url
        )
        # model names mapping
        self.model_map = {
            "sonar": "sonar",
            "sonar-pro": "sonar-pro",
            "sonar-reasoning-pro": "sonar-reasoning-pro",
            "common-model": "sonar-pro",
        }
        self.timeout = (10, 180)

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        try:
            model_id = self.get_model_name(model)
            if model_id is None:
                raise ValueError(f"Model {model} not available for provider.")
            first_token_time = None
            total_tokens = 0

            start = timer()
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                max_tokens=max_output,
                timeout=500
            )
            previous_completion_tokens = 0  # Initialize previous token count
            ttft = None
            elapsed = None
            first_chunk_tokens = 0

            response_list = []
            for chunk in response:
                response_list.append(json.loads(chunk.json()))
                if timer() - start > 90:
                    print("[WARN] Streaming exceeded 90s, stopping early.")
                    break
                usage = getattr(chunk, "usage", None)
                current_completion_tokens = (
                    getattr(usage, "completion_tokens", None)
                    if usage is not None else None
                )
                if current_completion_tokens is None:
                    current_completion_tokens = previous_completion_tokens
                new_tokens = current_completion_tokens - previous_completion_tokens
                previous_completion_tokens = current_completion_tokens
                if first_token_time is None:
                    first_token_time = timer()
                    ttft = first_token_time - start
                    first_chunk_tokens = max(new_tokens, 0)
                    if verbosity:
                        print(f"\nTime to First Token (TTFT): {ttft:.4f} seconds\n")

                total_tokens += max(new_tokens, 0)

                if getattr(chunk, "choices", None) and chunk.choices[0].finish_reason:
                    elapsed = timer() - start
                    if verbosity:
                        print(f"\nTotal Response Time: {elapsed:.4f} seconds")
                    break

                content = ""
                if getattr(chunk, "choices", None):
                    delta = getattr(chunk.choices[0], "delta", None)
                    content = getattr(delta, "content", "") if delta else ""
                if content:
                    print(content, end="")

            if elapsed is None:
                elapsed = timer() - start

            if ttft is None:
                ttft = elapsed
            non_first_latency = max(elapsed - ttft, 0.0)
            subsequent_tokens = max(total_tokens - first_chunk_tokens, 0)
            avg_tbt = (non_first_latency / subsequent_tokens) if subsequent_tokens > 0 else 0.0

            if verbosity:
                print(
                    f"\nNumber of output tokens: {total_tokens}, "
                    f"Time to First Token (TTFT): {ttft:.4f} seconds, "
                    f"Total Response Time: {elapsed:.4f} seconds, "
                    f"Avg TBT: {avg_tbt:.4f} seconds"
                )
            # Log metrics
            self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", elapsed)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "tps", total_tokens / elapsed if elapsed > 0 else 0)
            return response_list

        except Exception as e:
            print(f"[ERROR] Streaming inference failed for model '{model}': {e}")
            return e

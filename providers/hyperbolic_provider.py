import os
from openai import OpenAI
from providers.base_provider import BaseProvider


class Hyperbolic(BaseProvider):
    def __init__(self):
        """
        Initializes the HYPERBOLIC with the necessary API key and client.
        """
        perplexity_api = os.environ.get("HYPERBOLIC_API")

        if not perplexity_api:
            raise ValueError(
                "HYPERBOLIC_API token must be provided as an environment variable."
            )

        client_class = OpenAI
        base_url = "https://api.hyperbolic.xyz/v1"

        super().__init__(
            api_key=perplexity_api, client_class=client_class, base_url=base_url
        )

        # model names mapping
        self.model_map = {
            "meta-llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
            "qwen2-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
            "meta-llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "common-model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "reasoning-model": ["openai/gpt-oss-120b"]
        }

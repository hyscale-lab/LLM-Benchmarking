import os
from openai import OpenAI
from providers.base_provider import BaseProvider


class Open_AI(BaseProvider):
    def __init__(self):
        """
        Initializes the OPENAI with the necessary API key and client.
        """
        open_ai_api = os.environ["OPEN_AI_API"]
        super().__init__(api_key=open_ai_api, client_class=OpenAI)
        # model names
        self.model_map = {
            "meta-llama-3.2-3b-instruct": "gpt-4o-mini",  # speculative: 8-40b
            "mistral-7b-instruct-v0.1": "gpt-4o",  # speculative: 200-1000b
            "meta-llama-3.1-70b-instruct": "gpt-4",  # 1000-1800b
            "common-model": "gpt-4o",
            "common-model-small": "gpt-4.1-nano-2025-04-14"
        }

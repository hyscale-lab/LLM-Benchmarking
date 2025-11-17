import os
from groq import Groq
from providers.base_provider import BaseProvider


class GroqProvider(BaseProvider):
    def __init__(self):
        """
        Initializes the GROQ with the necessary API key and client.
        """
        groq_api = os.environ["GROQ_API_KEY"]
        super().__init__(api_key=groq_api, client_class=Groq)

        # model names
        self.model_map = {
            "google-gemma-7b-it": "gemma-7b-it",
            "meta-llama-3.2-3b-instruct": "llama-3.2-3b-preview",
            "meta-llama-3.1-70b-instruct": "llama-3.3-70b-versatile",
            "common-model": "llama-3.3-70b-versatile",
            "reasoning-model": ["openai/gpt-oss-120b"]
        }

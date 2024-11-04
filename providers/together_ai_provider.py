import os
from together import Together
from providers.base_provider import BaseProvider

class TogetherAI(BaseProvider):
    def __init__(self):
        together_api = os.environ["TOGETHER_AI_API"]
        super().__init__(api_key=together_api, client_class=Together)

        # model names 
        self.model_map = {
            "google-gemma-2b-it": "google/gemma-2b-it",
            "meta-llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
            "mistral-7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
            "meta-llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "meta-llama-3.1-405b-instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"           
        }

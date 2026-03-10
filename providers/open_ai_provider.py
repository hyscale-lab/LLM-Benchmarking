import os
from openai import OpenAI
from providers.base_provider import BaseProvider
from timeit import default_timer as timer
import re


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
            "reasoning-model": ["o4-mini", "gpt-4o"]
        }

    def _chat_for_eval(self, model_id, messages):
        start = timer()
        try:
            resp = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_completion_tokens=10000,
                temperature=1,
            )
            elapsed = timer() - start

            text = ""
            try:
                choice = resp.choices[0]
                msg = getattr(choice, "message", None)
                text = (getattr(msg, "content", "") or getattr(choice, "text", "") or "")
            except Exception:
                text = ""

            m = re.search(r"(?im)^\s*answer:\s*([0-9]{1,3})\s*$", text)
            if m:
                text = m.group(1)

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

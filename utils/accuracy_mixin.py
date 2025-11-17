import json
import re
import numpy as np

class AccuracyMixin:
    def measure_accuracy(self, dataset, verbosity=False, metric_default=None, model_id=None):
        accuracy_prompt = """
        You are solving AIME-style problems. You may show concise reasoning.
        On the LAST line, write exactly: "ANSWER: <integer>"
        Only one integer on that final line, between 0 and 999. Do not add any other text after it.
        """.strip()

        exs = self._load_dataset(dataset)

        if model_id is None:
            raise ValueError("Model not available.")

        scores, latencies, tokens = [], [], []

        for ex in exs:
            q = ex.get("prompt") or ex.get("question") or ""
            if not q:
                scores.append(0.0)
                latencies.append(0.0)
                tokens.append(0)
                continue

            text, tok, elapsed = self._chat_for_eval(
                model_id,
                [{"role": "system", "content": accuracy_prompt},
                 {"role": "user", "content": q}],
            )

            pred_for_scoring = self._extract_final_answer(text)
            s = self._score_example(ex, pred_for_scoring, metric_default=metric_default)

            scores.append(float(s))
            latencies.append(float(elapsed))
            tokens.append(int(tok))
            if verbosity:
                print(f"[{ex.get('id', '?')}] score={s:.2f} time={elapsed:.2f}s tokens={tok}")

        acc = float(np.mean(scores)) if scores else 0.0
        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        avg_tok = float(np.mean(tokens)) if tokens else 0.0

        return {"model": model_id, "n": len(exs), "accuracy": acc,
                "avg_latency_s": avg_lat, "avg_completion_tokens": avg_tok}

    def _chat_for_eval(self, model_id: str, messages: list):
        """Providers must implement. Return (text:str, completion_tokens:int, elapsed:float)."""
        raise NotImplementedError

    @staticmethod
    def _load_dataset(dataset):
        if isinstance(dataset, str):
            with open(dataset, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        return list(dataset)

    @staticmethod
    def _norm(s: str) -> str:
        s = str(s or "").lower().strip()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _extract_final_answer(text):
        print(text)
        s = str(text or "")
        # 1) Try tagged final answer
        m = re.search(rf"{re.escape("ANSWER:")}\s*([-+]?\d+)\s*$", s, flags=re.I | re.M)
        if m:
            return m.group(1)
        # 2) Fallback: last integer in the whole text (more robust than 'first')
        m_all = re.findall(r"[-+]?\d+", s)
        return m_all[-1] if m_all else s.strip()

    @classmethod
    def _score_example(cls, ex, pred, metric_default=None):
        metric = (ex.get("metric") or (metric_default or "exact")).lower()
        gold = ex.get("answer", "")

        if metric == "numeric":
            def to_int(x):
                try:
                    return int(x)
                except Exception:
                    m = re.search(r"-?\d+", str(x))
                    return int(m.group()) if m else None
            a, b = to_int(pred), to_int(gold)
            return 1.0 if (a is not None and b is not None and a == b) else 0.0

        # fallback to exact string match
        return 1.0 if cls._norm(pred) == cls._norm(gold) else 0.0

import os
import time
from timeit import default_timer as timer
import json
from dotenv import load_dotenv
import boto3
from providers.provider_interface import ProviderInterface
from utils.accuracy_mixin import AccuracyMixin


class AWSBedrock(AccuracyMixin, ProviderInterface):
    def __init__(self):
        """
        Initializes the AWS Bedrock client with credentials from environment variables.
        """
        load_dotenv()
        super().__init__()

        # model names
        self.model_map = {
            "meta-llama-3-70b-instruct": "meta.llama3-70b-instruct-v1:0",
            "common-model": "meta.llama3-70b-instruct-v1:0",
            "reasoning-model": ["us.anthropic.claude-3-7-sonnet-20250219-v1:0"]
        }

    def initialize_client(self):
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_BEDROCK_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_BEDROCK_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_BEDROCK_REGION"),
        )

    def get_model_name(self, model):
        return self.model_map.get(model, None)  # or model
    
    def normalize_messages(self, messages):
        if isinstance(messages, str):
            normalized_msgs = [{
                "role": "user",
                "content": [{"text": messages}]
            }]
        elif isinstance(messages, list):
            normalized_msgs = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role in ["user", "assistant"]:
                    normalized_msgs.append({
                        "role": role,
                        "content": [{"text": content}]
                    })
                else:
                    print(f"Invalid role found in messages: {role}")

        return normalized_msgs
    
    def construct_text_response(self, raw_response):
        if isinstance(raw_response, dict):
            text_response = raw_response['output']['message']['content'][0]['text']
        elif isinstance(raw_response, list):
            text_response = "".join(
                block['contentBlockDelta']['delta']['text']
                for block in raw_response
                if 'contentBlockDelta' in block
            )

        return text_response

    def perform_inference(self, model, messages, max_output=100, verbosity=True):
        """
        Performs a single-prompt inference using AWS Bedrock.
        """

        print("[INFO] Performing inference...")
        model_id = self.get_model_name(model)

        try:
            start_time = time.perf_counter()
            response = self.bedrock_client.converse(
                modelId=model_id,
                messages=self.normalize_messages(messages),
                system=[{"text": self.system_prompt}],
                inferenceConfig={
                    "maxTokens": max_output
                }
            )
            end_time = time.perf_counter()
            total_time = end_time - start_time
            self.log_metrics(model, "response_times", total_time)

            output_response = response['output']
            usage = response['usage']
            total_tokens = usage.get('outputTokens', 0)
            generated_text = output_response['message']['content'][0]['text']

            tbt = total_time / max(total_tokens - 1, 1)
            tps = (total_tokens / total_time)

            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)

            if verbosity:
                print(f"[INFO] Total response time: {total_time:.4f} seconds")
                print(f"[INFO] Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                print("[INFO] Generated response:")
                print(generated_text)

            return response

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return e

    def perform_inference_streaming(
        self, model, messages, max_output=100, verbosity=True
    ):
        """
        Performs a streaming inference using AWS Bedrock.
        """
        print("[INFO] Performing streaming inference...")

        model_id = self.get_model_name(model)

        inter_token_latencies = []
        first_token_time = None
        ttft = None
        start_time = time.perf_counter()
        try:
            response = self.bedrock_client.converse_stream(
                modelId=model_id,
                messages=self.normalize_messages(messages),
                system=[{"text": self.system_prompt}],
                inferenceConfig={
                    "maxTokens": max_output
                }
            )

            # Process the streaming response
            response_list = []
            for event in response["stream"]:
                response_list.append(event)

                if timer() - start_time > 90:
                    print("[WARN] Streaming exceeded 90s, stopping early.")
                    break

                if 'messageStop' in event:
                    stop_reason = event['messageStop']['stopReason']
                    if stop_reason == 'max_tokens':
                        print(f"\n[INFO] Stopped due to stop reason: {stop_reason}")
                        break

                if 'contentBlockDelta' in event:
                    chunk = event['contentBlockDelta']
                    current_token = chunk['delta']['text']

                    # Calculate timing
                    current_time = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = current_time
                        ttft = first_token_time - start_time
                        prev_token_time = first_token_time
                        print(
                            f"\n##### Time to First Token (TTFT): {ttft:.4f} seconds"
                        )
                        continue

                    # Capture token timing
                    time_to_next_token = time.perf_counter()
                    inter_token_latency = time_to_next_token - prev_token_time
                    prev_token_time = time_to_next_token
                    inter_token_latencies.append(inter_token_latency)
                    if verbosity:
                        if len(inter_token_latencies) < 20:
                            print(current_token, end="")  # Print the token
                        elif len(inter_token_latencies) == 21:
                            print("...")

            # Measure total response time
            total_time = time.perf_counter() - start_time

            token_count = len(inter_token_latencies) + (1 if ttft is not None else 0)
            non_first_latency = max(total_time - (ttft or 0.0), 0.0)
            avg_tbt = (non_first_latency / token_count) if token_count > 0 else 0.0

            if verbosity:
                print(f"\n##### Total Response Time: {total_time:.4f} seconds")
                print(f"##### Tokens: {token_count}")
                print(f"[INFO] Avg TBT ((total - TTFT)/tokens): {avg_tbt:.4f}s")

            self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            self.log_metrics(model, "totaltokens", token_count)
            self.log_metrics(model, "tps", (token_count / total_time) if total_time > 0 else 0.0)

            return response_list

        except Exception as e:
            print(f"[ERROR] Streaming inference failed: {e}")
            return e

    def _chat_for_eval(self, model_id, messages):
        system_prompt = None
        bedrock_messages = []

        for m in messages:
            role = m["role"]
            content = m.get("content", "")
            if role == "system":
                system_prompt = content
            elif role in ("user", "assistant"):
                bedrock_messages.append({"role": role, "content": content})

        # Claude Messages API format
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "max_tokens": 10000,
            "temperature": 0.0,
            "messages": bedrock_messages,
        })

        start = time.perf_counter()
        try:
            resp = self.bedrock_client.invoke_model(
                modelId=f"arn:aws:bedrock:us-east-1:356764711652:inference-profile/{model_id}",
                body=body,
            )
            # print(resp)
            elapsed = time.perf_counter() - start

            payload = json.loads(resp["body"].read())

            # Extract the full text from the assistant response
            text = ""
            if "content" in payload and isinstance(payload["content"], list):
                text = "".join([c.get("text", "") for c in payload["content"]])

            # Fallback token count if usage is provided
            tokens = int(payload.get("usage", {}).get("output_tokens", 0))

            return text, tokens, elapsed

        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"[ERROR] _chat_for_eval failed (Bedrock): {e}")
            return "", 0, float(elapsed)


# Example Usage
if __name__ == "__main__":
    aws_bedrock = AWSBedrock()
    model = "common-model"
    prompt = "Tell me a story."

    # Single-prompt inference
    generated_text, total_time = aws_bedrock.perform_inference(
        model=model, prompt=prompt, max_output=100, verbosity=True
    )

    # Streaming inference
    total_time, inter_token_latencies = aws_bedrock.perform_inference_streaming(
        model=model, prompt=prompt, max_output=10, verbosity=True
    )

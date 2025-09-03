import boto3
import os
import time
import json
import numpy as np
from dotenv import load_dotenv
from providers.provider_interface import ProviderInterface


class AWSBedrock(ProviderInterface):
    def __init__(self):
        """
        Initializes the AWS Bedrock client with credentials from environment variables.
        """
        load_dotenv()
        super().__init__()

        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=os.getenv("AWS_BEDROCK_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_BEDROCK_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_BEDROCK_REGION"),
        )

        # model names
        self.model_map = {
            "meta-llama-3-70b-instruct": "meta.llama3-70b-instruct-v1:0",
            "common-model": "meta.llama3-70b-instruct-v1:0",
        }

    def get_model_name(self, model):
        return self.model_map.get(model, None)  # or model

    def format_prompt(self, user_prompt):
        """
        Combines the system prompt and user prompt into a single formatted prompt.
        """
        return f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {self.system_prompt}
        <|start_header_id|>user<|end_header_id|>
        {user_prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

    def perform_inference(self, model, prompt, max_output=100, verbosity=True):
        """
        Performs a single-prompt inference using AWS Bedrock.
        """

        print("[INFO] Performing inference...")
        model_id = self.get_model_name(model)
        formatted_prompt = self.format_prompt(prompt)
        print(formatted_prompt)
        # Prepare the request payload
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": max_output,
        }
        request_body = json.dumps(native_request)

        try:
            start_time = time.perf_counter()
            response = self.bedrock_client.invoke_model(
                modelId=model_id, body=request_body
            )
            end_time = time.perf_counter()
            total_time = end_time - start_time
            self.log_metrics(model, "response_times", total_time)

            model_response = json.loads(response["body"].read())
            generated_text = model_response.get("generation", "")

            total_tokens = model_response.get("generation_token_count")

            tbt = total_time / max(total_tokens, 1)
            tps = (total_tokens / total_time) if total_time > 0 else 0.0

            self.log_metrics(model, "totaltokens", total_tokens)
            self.log_metrics(model, "timebetweentokens", tbt)
            self.log_metrics(model, "tps", tps)

            if verbosity:
                print(f"[INFO] Total response time: {total_time:.4f} seconds")
                print(f"[INFO] Tokens: {total_tokens}, Avg TBT: {tbt:.4f}s, TPS: {tps:.2f}")
                print("[INFO] Generated response:")
                print(generated_text)

            return generated_text, total_time

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return None, None

    def perform_inference_streaming(
        self, model, prompt, max_output=100, verbosity=True
    ):
        """
        Performs a streaming inference using AWS Bedrock.
        """
        print("[INFO] Performing streaming inference...")

        model_id = self.get_model_name(model)

        # Prepare the request payload
        formatted_prompt = self.format_prompt(prompt)

        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": max_output,
        }
        request_body = json.dumps(native_request)

        inter_token_latencies = []
        first_token_time = None
        ttft = None
        start_time = time.perf_counter()
        try:
            streaming_response = self.bedrock_client.invoke_model_with_response_stream(
                modelId=model_id, body=request_body
            )

            # Process the streaming response
            for event in streaming_response["body"]:
                if event:
                    try:
                        # print(f"[DEBUG] {event}")
                        chunk = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                        # print(chunk)
                    except Exception:
                        # print(f"[DEBUG] Failed to decode chunk: {e}")
                        continue
                    
                    if chunk["stop_reason"] == 'length':
                        total_time = time.perf_counter() - start_time
                        print(chunk)
                        break

                    if "generation" in chunk:
                        current_token = chunk["generation"]

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
            if verbosity:
                print(f"\n##### Total Response Time: {total_time:.4f} seconds")
                print(f"##### Tokens: {len(inter_token_latencies)}")
                avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
                median = np.percentile(inter_token_latencies, 50)
                p95 = np.percentile(inter_token_latencies, 95)
                print("[INFO] avg_tbt - ", avg_tbt, median, p95)
            self.log_metrics(model, "timetofirsttoken", ttft)
            self.log_metrics(model, "response_times", total_time)
            self.log_metrics(model, "timebetweentokens", avg_tbt)
            # print(median, p95)
            self.log_metrics(model, "timebetweentokens_median", median)
            self.log_metrics(model, "timebetweentokens_p95", p95)
            self.log_metrics(model, "totaltokens", len(inter_token_latencies) + 1)
            self.log_metrics(
                model, "tps", (len(inter_token_latencies) + 1) / total_time
            )

            return total_time, inter_token_latencies

        except Exception as e:
            print(f"[ERROR] Streaming inference failed: {e}")
            return None, None


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

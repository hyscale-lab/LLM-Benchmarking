import boto3
import os
import time
import json
import numpy as np
from dotenv import load_dotenv
from providers.provider_interface import ProviderInterface
import math


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
            "meta-llama-3-8b-instruct": "meta.llama3-8b-instruct-v1:0",
            "mistral-48b-instruct-v0.1": "mistral.mixtral-8x7b-instruct-v0:1",
            "mistral-23b-instruct-v0.1": "mistral.mistral-small-2402-v1:0",
            "mistral-124b-instruct-v0.1": "mistral.mistral-large-2402-v1:0",
            "deepseek-r1": "us.deepseek.r1-v1:0",
            "common-model": "meta.llama3-70b-instruct-v1:0",
            "common-model-small": "meta.llama3-1-8b-instruct-v1:0"
        }

    def get_model_name(self, model):
        return self.model_map.get(model, None)  # or model

    def get_model_provider(self, model_id):
        return model_id[:model_id.find('.')]

    def format_prompt(self, user_prompt):
        """
        Combines the system prompt and user prompt into a single formatted prompt.
        """
        return f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an obedient assistant.
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
        print(max_output)
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

            if verbosity:
                print(f"[INFO] Total response time: {total_time:.4f} seconds")
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
        model_provider = self.get_model_provider(model_id)

        system_messages = [
            {
                "text": "You are a helpful assistant."
            }
        ]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]

        # Prepare the request payload
        print(model_provider)
        formatted_prompt = self.format_prompt(prompt)
        # if model_provider == "meta":
        #     max_tokens_config = 'max_gen_len'
        # elif model_provider == "us":
        #     max_tokens_config = 'maxTokens'
        #     # max_output = max_output + len(prompt)
        # else:
        #     max_tokens_config = 'max_tokens'

        native_request = {
            # "prompt": formatted_prompt,
            # max_tokens_config: max_output,
            "maxTokens": max_output,
        }
        print(max_output)
        request_body = json.dumps(native_request)

        inter_token_latencies = []
        first_token_time = None
        ttft = None
        total_tokens = 0
        c = 0
        # print(prompt)
        start_time = time.perf_counter()
        try:
            # streaming_response = self.bedrock_client.invoke_model_with_response_stream(
            #     modelId="us.deepseek.r1-v1:0", body=request_body
            # )
            
            streaming_response = self.bedrock_client.converse_stream(
                modelId=model_id,
                system=system_messages,
                inferenceConfig=native_request,
                messages=messages
            )

            print(streaming_response)
            for event in streaming_response['stream']:
                
                # print(f"{event}")
                if 'metadata' in event:
                    # print(event['metadata']['usage']['outputTokens'])
                    print(event['metadata']['usage'])
                    total_tokens = event['metadata']['usage']['outputTokens']
                    total_time = time.perf_counter() - start_time
                    print(f"Total time: {total_time}s. Total tokens: {total_tokens}, {c}, {len(inter_token_latencies)}.")
                    break
                if "contentBlockDelta" in event:
                    delta = event['contentBlockDelta']['delta']

                    if "SDK_UNKNOWN_MEMBER" in delta:
                        real_tag = delta["SDK_UNKNOWN_MEMBER"]["name"]
                        # print(real_tag)
                        # If the tag is 'reasoningContent', extract its nested 'text'
                        if real_tag == "reasoningContent":
                            # see the content in reasoningContent HOW?
                            # if "text" in delta["SDK_UNKNOWN_MEMBER"]["name"]:
                            #     print("??")
                            # else:
                            #     print("....")
                            
                            if first_token_time is None:
                                current_time = time.perf_counter()
                                first_token_time = current_time
                                ttft = first_token_time - start_time
                                prev_token_time = first_token_time
                                print(
                                    f"\n##### Time to First Token (TTFT): {ttft:.4f} seconds"
                                )
                                continue

                            time_to_next_token = time.perf_counter()
                            inter_token_latency = time_to_next_token - prev_token_time
                            prev_token_time = time_to_next_token
                            inter_token_latencies.append(inter_token_latency)

                    if "text" in delta:
                        text = delta["text"]
                        
                        print(f"{text}", end="")
                
                        if first_token_time is None:
                            current_time = time.perf_counter()
                            first_token_time = current_time
                            ttft = first_token_time - start_time
                            prev_token_time = first_token_time
                            print(
                                f"\n##### Time to First Token (TTFT): {ttft:.4f} seconds"
                            )
                            continue

                        time_to_next_token = time.perf_counter()
                        # inter_token_latency = time_to_next_token - prev_token_time
                        elapsed = time_to_next_token - prev_token_time
                        token_count = len(text.split())
                        c += token_count
                        avg_latency = elapsed / max(1, token_count)
                        # print(f"Len: {token_count}, {text}", end="~")

                        # record one latency entry *per* token
                        inter_token_latencies.extend([avg_latency] * token_count)
                        prev_token_time = time_to_next_token
                        # inter_token_latencies.append(inter_token_latency)

                            
            # Process the streaming response
            # for event in streaming_response["body"]:
            #     if event:
            #         # print(event)
            #         try:
            #             # print(f"[DEBUG] {event}")
            #             chunk = json.loads(event["chunk"]["bytes"].decode("utf-8"))
            #             # print(chunk)
            #         except Exception:
            #             # print(f"[DEBUG] Failed to decode chunk: {e}")
            #             continue
                    
            #         if chunk.get("stop_reason") == "length" or chunk.get("outputs", [{}])[0].get("stop_reason") == "length":
            #             total_time = time.perf_counter() - start_time
            #             # print(chunk)
            #             total_tokens = chunk['generation_token_count']
            #             print(f"Total tokens (real): {chunk['generation_token_count']}")
            #             break
                    
            #         if "outputs" in chunk or 'generation' in chunk:
            #             if "outputs" in chunk :
            #                 current_token = chunk["outputs"][0]['text']
            #             if 'generation' in chunk:
            #                 current_token = chunk["generation"]
            #             # print(current_token)
            #             # Calculate timing
            #             current_time = time.perf_counter()
            #             if first_token_time is None:
            #                 first_token_time = current_time
            #                 ttft = first_token_time - start_time
            #                 prev_token_time = first_token_time
            #                 print(
            #                     f"\n##### Time to First Token (TTFT): {ttft:.4f} seconds"
            #                 )
            #                 continue

            #             # Capture token timing
            #             time_to_next_token = time.perf_counter()
            #             inter_token_latency = time_to_next_token - prev_token_time
            #             prev_token_time = time_to_next_token
            #             inter_token_latencies.append(inter_token_latency)
            #             print(current_token, end=" | ")

                        # if verbosity:
                        #     if len(inter_token_latencies) < 20:
                        #         print(current_token, end="")
                        #     elif len(inter_token_latencies) == 21:
                        #         print("...")
    

            # Measure total response time
            # total_time = time.perf_counter() - start_time
            if verbosity:
                print(f"\n##### Total Response Time: {total_time:.4f} seconds")
                print(f"Total tokens: {total_tokens}, {c}, {len(inter_token_latencies)}.")
                avg_tbt = sum(inter_token_latencies) / len(inter_token_latencies)
                median = np.percentile(inter_token_latencies, 50)
                p95 = np.percentile(inter_token_latencies, 95)
                # print("[INFO] avg_tbt - ", avg_tbt, median, p95, len(inter_token_latencies))
            # print(prompt, 10 ** math.ceil(math.log10(10 ** math.ceil(math.log10(len(prompt.split(" ")))))))
            self.log_metrics(model_name=model, input_size=10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output=max_output, metric="timetofirsttoken", value=ttft)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "response_times", total_time)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens", avg_tbt)
            # print(median, p95)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_median", median)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "timebetweentokens_p95", p95)
            self.log_metrics(model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "totaltokens", total_tokens)
            self.log_metrics(
                model, 10 ** math.ceil(math.log10(len(prompt.split(" ")))), max_output, "tps", (len(inter_token_latencies) + 1) / total_time
            )

            return total_time, inter_token_latencies

        except Exception as e:
            print(f"[ERROR] Streaming inference failed: {e}")
            return None, None


# Example Usage
if __name__ == "__main__":
    aws_bedrock = AWSBedrock()
    model = "common-model"
    prompt = "Write a standalone narrative of exactly 100 000 tokens: a complete, self-contained fantasy saga with clear chapters, rising action, characters, dialogue, and resolution. Do not add any extra explanation—produce only the story text until you reach 100 000 tokens exact."

    # Single-prompt inference
    # generated_text, total_time = aws_bedrock.perform_inference(
    #     model=model, prompt=prompt, max_output=100, verbosity=True
    # )

    # Streaming inference
    total_time, inter_token_latencies = aws_bedrock.perform_inference_streaming(
        model=model, prompt=prompt, max_output=10000, verbosity=True
    )
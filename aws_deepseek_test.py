import boto3
import os
from dotenv import load_dotenv

load_dotenv()

bedrock_rt = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_BEDROCK_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_BEDROCK_SECRET_ACCESS_KEY"),
    region_name="us-west-2",
)


# Define the system prompt...
system_messages = [{
"text": "You're a helpful assistant"
}
]
# Define the user prompt...
messages = [ 
    {
        "role": "user",
        "content": [ 
            {
                "text": "Tell me why 2 + 2 = 4, like I'm an alien."
            }
        ]
    }
]


streaming_response = bedrock_rt.converse_stream(
    modelId="us.deepseek.r1-v1:0" ,
    inferenceConfig= {
        "maxTokens": 30000,
        "temperature": 1
    },
    system=system_messages, 
    messages=messages
)

# Process the streaming response...
for chunk in streaming_response["stream"]: 
    print(chunk)
    # if "contentBlockDelta" in chunk:
    #     delta = chunk ["contentBlockDelta"] ["delta"]
    # # Handle reasoning content (displayed in green)
    # if "reasoningContent" in delta:
    #     if "text" in delta [" reasoningContent"]:
    #         reasoning_text = delta ["reasoningContent"] ["text"]
    #         print("\033[92m" + reasoning_text + "\033[0m", end="**")
    #     else:
    #         print("**")
    # # Handle regular text content (displayed in white)
    # if "text" in delta:
    #     text = delta ["text"]
    #     print(text, end="**")
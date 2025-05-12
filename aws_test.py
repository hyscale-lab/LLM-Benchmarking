# Use the ListFoundationModels API to show the models that are available in your region.
import boto3
import os
from dotenv import load_dotenv

load_dotenv()
             
# Create an &BR; client in the &region-us-east-1; Region.
# print(os.getenv('AWS_BEDROCK_REGION'), os.getenv('GEMINI_API_KEY'))
bedrock = boto3.client(
    service_name="bedrock",
    aws_access_key_id=os.getenv('AWS_BEDROCK_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_BEDROCK_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_BEDROCK_REGION')
)

response = bedrock.list_foundation_models()

# Extract model summaries
models = response.get("modelSummaries", [])

# Filter models that support ON_DEMAND inference
on_demand_models = [
    {
        "modelName": model["modelName"],
        "modelId": model["modelId"],
        "modelArn": model["modelArn"],
        "providerName": model["providerName"]
    }
    for model in models
    if "ON_DEMAND" in model.get("inferenceTypesSupported", []) and model.get("responseStreamingSupported")
]

# Print the filtered models
for model in on_demand_models:
    print(f"Model Name: {model['modelName']}")
    print(f"Model ID: {model['modelId']}")
    print(f"Model ARN: {model['modelArn']}")
    print(f"Provider Name: {model['providerName']}")
    print()

print(len(models), len(on_demand_models))

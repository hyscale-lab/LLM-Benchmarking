# **LLMetrics: Benchmarking LLM Inference Services**

LLMetrics is a comprehensive benchmarking tool designed to evaluate and compare the performance of Large Language Model (LLM) inference APIs across various providers. It measures key metrics such as Time-to-First-Token (TTFT), Time-Between-Tokens (TBT), and overall End-to-End (E2E) latency in a standardized testing environment.

## **Features**

* Standardized Testing: Uses fixed prompts, input tokens, and output tokens for consistent performance evaluation
* Provider Comparison: Benchmarks multiple LLM service providers, including cloud APIs (e.g., OpenAI, Anthropic, Cloudflare) and local servers (e.g., vLLM)
* Configurable Experiments: Define experiments via a JSON configuration file that specifies providers, models, number of requests, token sizes, and streaming mode
* Data Persistence: Stores benchmarking results in DynamoDB for scalable, historical tracking
* Visualization: Generates latency and CDF plots which are integrated into an interactive dashboard for actionable insights
* Automated Workflows: Scheduled experiments (e.g., weekly via AWS VM and GitHub Actions) ensure continuous performance monitoring

## **Setup**

### **1. Clone the Repository**

```
# Using HTTPS
git clone https://github.com/your-username/LLMetrics.git

# Using SSH
git clone git@github.com:your-username/LLMetrics.git

cd LLMetrics
```

### **2\. Install Dependencies**

`pip install \-r requirements.txt`

### **3\. Configure Environment Variables**

Create a .env file in the repository root with your API keys and credentials:

```
AWS_REGION=your_aws_region
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
CLOUDFLARE_ACCOUNT_ID="your-cloudflare-account-id"
CLOUDFLARE_AI_TOKEN="your-cloudflare-ai-token"
TOGETHER_AI_API="your-together-ai-api-key"
OPEN_AI_API="your-openai-api-key"
ANTHROPIC_API="your-anthropic-api-key"
PERPLEXITY_AI_API="your-perplexity-ai-api-key"
HYPERBOLIC_API="your-hyperbolic-api-key"
GROQ_API_KEY="your-groq-api-key"
GEMINI_API_KEY="your-gemini-api-key"
MISTRAL_LARGE_API="your-mistral-large-api-key"
AWS_BEDROCK_ACCESS_KEY_ID="your-aws-bedrock-access-key-id"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-bedrock-secret-key"
AWS_BEDROCK_REGION="your-aws-bedrock-region"
DYNAMODB_ENDPOINT_URL="your-dynamodb-endpoint-url"
AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
AZURE_AI_ENDPOINT="your-azure-ai-endpoint"
AZURE_AI_API_KEY="your-azure-ai-api-key"
```
## **Usage**

### **1. Create a Configuration File**

Create a config.json file to define your benchmarking experiment:
```
{ "providers": 
    [ "TogetherAI", "Cloudflare", "OpenAI", "Anthropic", "vLLM" \], 
  "models": 
    [ "common-model" ], 
  "num_requests": 100, 
  "input_tokens": 10, 
  "streaming": true,
  "input_type": "static",  # or "trace"
  "max_output": 100, 
  "verbose": true,
  "dataset": "aime.jsonl"
}
```
### **2. Run the Benchmark**

```
python main.py -c config.json`
OR
python main.py -c config.json --vllm_ip <host ip>
```

### **3. View Results**

LLMetrics saves plots (latency graphs - CDF Plots) in the designated output directory `benchmark_graph`

## **Input Type**

LLMetrics supports these input types, which can be set in the consiguration file using `input_type`. This does not apply to `Accuracy` metrics since it uses its own input.
- `static`: Use the same prompt for every request.
- `trace`: Use a preprocessed input derived from Azure trace dataset. See [releases](https://github.com/hyscale-lab/LLM-Benchmarking/releases).

## **Continuous Benchmarking Workflow**

LLMetrics integrates with a CI/CD pipeline to run weekly experiments on an AWS VM:

* GitHub Actions: Automates weekly benchmarking runs and CI tests  
* AWS VM: Provides a stable network environment for consistent benchmarking  
* DynamoDB: Stores benchmark data securely for historical analysis  
* Dashboard: An interactive dashboard deployed on GitHub Pages visualizes the results


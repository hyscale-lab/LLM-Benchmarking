# **LLMetrics: Benchmarking LLM Inference Services**

LLMetrics is a comprehensive benchmarking tool designed to evaluate and compare the performance of Large Language Model (LLM) inference APIs across various providers. It measures key metrics such as Time-to-First-Token (TTFT), Time-Between-Tokens (TBT), and overall End-to-End (E2E) latency in a standardized testing environment.

## **Features**

* **Standardized Testing**: Uses fixed prompts, input tokens, and output tokens for consistent performance evaluation
* **Provider Comparison**: Benchmarks 10 LLM service providers — Anthropic, AWS Bedrock, Azure, Cloudflare, Google, Groq, Hyperbolic, OpenAI, PerplexityAI, TogetherAI
* **Multiple Input Types**: Supports static, trace, multiturn, and VQA (Vision Question Answering) benchmarks
* **Prompt Caching**: Measures cache read/write token savings in multiturn conversations for providers that support it (Anthropic, AWS Bedrock, Azure, Google, OpenAI, Groq)
* **Configurable Experiments**: Define experiments via a JSON configuration file that specifies providers, models, number of requests, token sizes, and streaming mode
* **Visualization**: Generates latency and CDF plots integrated into an interactive dashboard for actionable insights
* **Automated Workflows**: Scheduled experiments via GitHub Actions ensure continuous performance monitoring

## **Setup**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/LLMetrics.git
cd LLMetrics
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Configure Environment Variables**

Create a `.env` file in the repository root with your API keys and credentials:

```
# AWS Bedrock
AWS_BEDROCK_ACCESS_KEY_ID="your-aws-bedrock-access-key-id"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-bedrock-secret-key"
AWS_BEDROCK_REGION="your-aws-bedrock-region"

# Azure
AZURE_AI_ENDPOINT="your-azure-ai-endpoint"
AZURE_AI_API_KEY="your-azure-ai-api-key"
AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"

# Cloudflare
CLOUDFLARE_ACCOUNT_ID="your-cloudflare-account-id"
CLOUDFLARE_AI_TOKEN="your-cloudflare-ai-token"

# Google
GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
GOOGLE_CLOUD_PROJECT="your-google-cloud-project"
GOOGLE_CLOUD_LOCATION="your-google-cloud-location"
LLAMA_4_GOOGLE_CLOUD_LOCATION="your-llama4-specific-region"

# Anthropic / Groq / Hyperbolic / OpenAI / PerplexityAI / TogetherAI
ANTHROPIC_API="your-anthropic-api-key"
GROQ_API_KEY="your-groq-api-key"
HYPERBOLIC_API="your-hyperbolic-api-key"
OPEN_AI_API="your-openai-api-key"
PERPLEXITY_AI_API="your-perplexity-ai-api-key"
TOGETHER_AI_API="your-together-ai-api-key"

# HuggingFace (required for VQA tokenizer downloads)
HF_TOKEN="your-huggingface-token"
```

When `backend: true` is set in the config (see below), results are persisted to DynamoDB. This requires additional AWS credentials:

```
AWS_ACCESS_KEY_ID="your-aws-access-key-id"
AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
AWS_REGION="your-aws-region"
```

## **Usage**

### **1. Create a Configuration File**

Create a `config.json` to define your benchmarking experiment:

```json
{
  "providers": [
    "Anthropic",
    "AWSBedrock",
    "Azure",
    "Cloudflare",
    "Google",
    "Groq",
    "Hyperbolic",
    "OpenAI",
    "PerplexityAI",
    "TogetherAI"
  ],
  "models": ["common-model"],
  "input_type": "static",
  "num_requests": 100,
  "input_tokens": 10,
  "max_output": 100,
  "streaming": true,
  "verbose": true,
  "backend": false
}
```

Set `"input_type"` to `"static"`, `"trace"`, `"multiturn"`, or `"vqa"` depending on the benchmark. Set `"backend": true` to persist results to DynamoDB (requires AWS credentials above).

### **2. Run the Benchmark**

```bash
python main.py -c config.json

# With a local vLLM server
python main.py -c config.json --vllm_ip <host-ip>

# List all available providers and their model mappings
python main.py --list
```

### **3. View Results**

LLMetrics saves latency graphs and CDF plots to `benchmark_graph/`.

- **Multiturn** runs additionally produce per-turn token usage CSVs in `multiturn/logs/`
- **VQA** runs produce per-sample TTFT CSVs in `vqa/logs/`
- **Trace** runs produce a `.result` file per provider in `trace/`. Proxy traffic logging to `trace/proxy/traffic.log` exists in the codebase but is currently disabled.

## **Input Types**

Set via `input_type` in the configuration file. For datasets, see [releases](https://github.com/hyscale-lab/LLM-Benchmarking/releases).

| Type | Description |
|---|---|
| `static` | Same prompt repeated for every request |
| `trace` | Preprocessed inputs derived from the Azure trace dataset; a proxy server is started per provider run and a load generator replays requests at realistic arrival rates — detailed latency results are written to `trace/<Provider>.result`. Proxy traffic logging to `trace/proxy/traffic.log` exists in the codebase but is currently disabled. |
| `multiturn` | Multi-turn conversations derived from the ShareGPT dataset; supports prompt caching |
| `vqa` | Vision Question Answering — measures multimodal vs text-only TTFT to isolate vision encoder latency |

## **Models**

Each provider maps canonical model aliases to its own model IDs. Supported aliases:

| Alias | Purpose |
|---|---|
| `common-model` | Standard chat model for latency benchmarks |
| `cache-model` | Model used for prompt caching multiturn benchmarks |
| `reasoning-model` | Reasoning/thinking model |
| `vision-model-01` / `vision-model-02` | Vision models for VQA benchmarks |

## **Prompt Caching**

When `input_type` is `multiturn`, LLMetrics runs all configured providers regardless of whether they support caching. Providers that do support it use their native caching mechanism:

* **Anthropic / AWS Bedrock**: Explicit cache control markers placed on conversation history (two-phase: write on first turn confirmed by the API, read+write on subsequent turns)
* **Google**: Context cache object created from conversation history before each turn
* **Azure / OpenAI / Groq**: Automatic server-side prefix caching (no explicit markers needed)

For providers without caching support (Cloudflare, Hyperbolic, PerplexityAI, TogetherAI), the multiturn benchmark still runs and collects latency metrics — cache columns in the CSV log will be `0`.

Per-turn `cache_read` and `cache_write` token counts are recorded in `multiturn/logs/<Provider>_<model>.csv` alongside `total_input` and `output` tokens.

## **Continuous Benchmarking Workflow**

LLMetrics integrates with GitHub Actions to run scheduled experiments:

* **Monday**: Static benchmark
* **Tuesday**: Trace benchmarks (A + B)
* **Wednesday**: Multiturn benchmarks (A + B + C)
* **Thursday**: VQA benchmarks (A + B)

Results and logs are uploaded as artifacts and visualized in an interactive dashboard deployed on GitHub Pages.

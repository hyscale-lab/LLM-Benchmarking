"""Main module for running benchmarks on selected AI providers and models."""

import argparse
import json
import sys
from dotenv import load_dotenv
from providers import (
    TogetherAI,
    Cloudflare,
    Open_AI,
    GoogleGemini,
    GroqProvider,
    Anthropic,
    PerplexityAI,
    Hyperbolic,
    Azure,
    AWSBedrock,
    vLLM
)
# from utils.prompt_generator import get_prompt
from utils.db_utils import create_experiment_folder, save_config

# Load environment variables
load_dotenv()


# Define input parser
parser = argparse.ArgumentParser(
    description="Run a benchmark on selected AI providers and models.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-c", "--config", type=str, help="Path to the JSON configuration file"
)
parser.add_argument(
    "--list", action="store_true", help="List available providers and models"
)
parser.add_argument(
    "--vllm_ip", type=str, default=None, help="IP address of vLLM provider"
)

# Define possible input sizes
input_sizes = [10, 100, 1000, 10000, 100000]

# Define possible max output tokens
OUTPUT_SIZE_UPPER_LIMIT = 100000
OUTPUT_SIZE_LOWER_LIMIT = 100


def get_available_providers():
    """Returns a dictionary of available providers and their instances."""
    available_providers = {
        "TogetherAI": TogetherAI(),
        "Cloudflare": Cloudflare(),
        "OpenAI": Open_AI(),
        "PerplexityAI": PerplexityAI(),
        "Hyperbolic": Hyperbolic(),
        "Google": GoogleGemini(),
        "Anthropic": Anthropic(),
        "Groq": GroqProvider(),
        "Azure": Azure(),
        "AWSBedrock": AWSBedrock(),
        "vLLM": vLLM()
    }

    return available_providers


# Function to load JSON configuration
def load_config(file_path):
    """
    Loads JSON configuration from the specified file path.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to parse the configuration file. Ensure it is valid JSON.")
        return None


# Function to display available providers and their models
def display_available_providers():
    """Displays available providers and their models."""
    print("\nAvailable Providers and Models:")
    for provider_name, provider_instance in get_available_providers().items():
        print(f"\n{provider_name}")
        if hasattr(provider_instance, "model_map"):
            for common_name, model_name in provider_instance.model_map.items():
                print(f"  - {common_name}: {model_name}")

        else:
            print("  No models available.")


# Function to validate provider selection
def validate_providers(selected_providers):
    """Validates selected providers and returns a list of provider instances."""
    valid_providers = []
    for provider_name in selected_providers:
        if provider_name in get_available_providers():
            valid_providers.append(get_available_providers()[provider_name])
        else:
            # logging.warning(f"Warning: {provider_name} is not a valid provider name.")
            print(f"Warning: {provider_name} is not a valid provider name.")
    return valid_providers


# Function to get common models across selected providers
def get_common_models(selected_providers):
    """Returns a list of common models across the selected providers."""
    model_sets = []
    for provider in selected_providers:
        if hasattr(provider, "model_map"):
            models = set(provider.model_map.keys())  # Fetch model names from model_map
            model_sets.append(models)

    common_models = set.intersection(*model_sets) if model_sets else set()
    return list(common_models)


# Validate user-selected models
def validate_selected_models(selected_models, common_models, selected_providers):
    """
    Validates user-selected models and returns a list of valid models.
    """
    valid_models = []
    for model in selected_models:
        if model in common_models:
            valid_models.append(model)
        else:
            if len(selected_providers) > 1:
                print(
                    f"Warning: Model '{model}' is not a common model among the chosen providers. \
                Please select common models."
                )
            else:
                for provider in selected_providers:
                    if model in provider.model_map:
                        valid_models.append(model)
                        break
                    print(
                        f"Warning: Model '{model}' not available for all selected providers."
                    )
    return valid_models


# Main function to run the benchmark
def run_benchmark(config, vllm_ip=None):
    # exp_id, exp_dir = create_experiment_folder("/home/users/ntu/kavi0008/loadgen/experiments/aws")
    exp_id, exp_dir = create_experiment_folder("experiments/togetherai_4_6000_merge")
    print(exp_id, exp_dir)
    save_config(config, exp_dir)
    """Runs the benchmark based on the given configuration."""
    providers = config.get("providers", [])
    num_requests = config.get("num_requests", 1)
    models = config.get("models", [])
    # input_tokens = config.get("input_tokens", 10)
    input_tokens = config.get("input_tokens", [10])
    streaming = config.get("streaming", False)
    # max_output = config.get("max_output", 100)
    outputs = config.get("max_output", [100])
    verbose = config.get("verbose", False)
    backend = config.get("backend", False)
    dataset = config.get("dataset", "general")

    # Select Benchmark class based on backend flag
    if backend:
        from benchmarking.dynamo_bench import Benchmark
    else:
        from benchmarking.benchmark_main import Benchmark
    # Validate and initialize providers
    selected_providers = validate_providers(providers)
    print(
        f"Selected Providers: {[provider.__class__.__name__ for provider in selected_providers]}"
    )

    # Get common models from selected providers
    common_models = (
        get_common_models(selected_providers) if len(selected_providers) > 1 else []
    )
    if not common_models and len(selected_providers) > 1:
        # logging.error("No common models found among selected providers.")
        print("No common models found among selected providers.")
        return

    # Validate models
    valid_models = validate_selected_models(models, common_models, selected_providers)
    if not valid_models:
        print(
            "No valid/common models selected. Ensure models are available across providers."
        )
        display_available_providers()
        return

    # logging.info(f"Selected Models: {valid_models}")
    print(f"Selected Models: {valid_models}")

    # handling input tokens
    for input_token_size in input_tokens:
        if input_token_size not in input_sizes:
            print(f"Please enter an input token from the following choices: {input_sizes}")
            return

    # prompts = []
    # for input_token_size in input_tokens:
    #     prompt = get_prompt(input_token_size)
    #     prompts.append(prompt)
    # print(f"Prompt: {prompt}")

    for max_output in outputs:
        if max_output < OUTPUT_SIZE_LOWER_LIMIT or max_output > OUTPUT_SIZE_UPPER_LIMIT:
            print(
                f"Please enter an output token length between \
                {OUTPUT_SIZE_LOWER_LIMIT} and {OUTPUT_SIZE_UPPER_LIMIT}."
            )
            return

    print("\nRunning benchmark...")
    benchmark = Benchmark(
        selected_providers,
        num_requests,
        valid_models,
        outputs,
        # prompts=prompts,
        inputs=input_tokens,
        dataset=dataset,
        streaming=streaming,
        verbosity=verbose,
        vllm_ip=vllm_ip,
        exp_dir=exp_dir
    )
    benchmark.run()


def main():
    """Main function to parse arguments and run the program."""
    input_data = sys.stdin.read()
    request = json.loads(input_data)
    print(request)
    args = parser.parse_args()
    
    vllm_ip = getattr(args, "vllm_ip", None)
    # Display available providers and models if --list flag is used
    if args.list:
        display_available_providers()
    elif request:
        config = request['context']
        print(config)
        if config:
            if "vLLM" in config.get("providers", []):
                vllm_ip = request['vllm_ip']
                print(vllm_ip)
                if not vllm_ip:
                    print("\n[ERROR] vLLM provider is selected, but `vllm_ip` is missing!")
                    print("   ➜ Please add `vllm_ip' via CLI using `--vllm_ip <ip-addr>`.")
                    return  # Stop execution
        
            run_benchmark(config, vllm_ip)
    elif args.config:
        config = load_config(args.config)
        if config:
            if "vLLM" in config.get("providers", []) and not vllm_ip:
                print("\n[ERROR] vLLM provider is selected, but `vllm_ip` is missing!")
                print("   ➜ Please add `vllm_ip' via CLI using `--vllm_ip <ip-addr>`.")
                return  # Stop execution
        
            run_benchmark(config, vllm_ip)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
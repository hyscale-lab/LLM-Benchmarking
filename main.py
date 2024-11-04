"""Main module for running benchmarks on selected AI providers and models."""

import argparse
import json
from dotenv import load_dotenv
from benchmarking.benchmark_main import Benchmark
from providers import TogetherAI, Cloudflare, Open_AI
from utils.prompt_generator import get_prompt

# Load environment variables
load_dotenv()

# Define input parser
parser = argparse.ArgumentParser(
    description="Run a benchmark on selected AI providers and models.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-c", "--config", type=str, help="Path to the JSON configuration file")
parser.add_argument("--list", action="store_true", help="List available providers and models")

# Define possible input sizes
input_sizes = [10, 100, 1000, 10000, 100000]

# Define possible max output tokens
OUTPUT_SIZE_UPPER_LIMIT = 5000
OUTPUT_SIZE_LOWER_LIMIT = 100

def get_available_providers():
    """Returns a dictionary of available providers and their instances."""
    available_providers = {
        "TogetherAI": TogetherAI(),
        "Cloudflare": Cloudflare(),
        "OpenAI": Open_AI(),
        # "PerplexityAI": PerplexityAI(),
    }

    return available_providers

# Function to load JSON configuration
def load_config(file_path):
    """Loads JSON configuration from the specified file path."""    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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
        if hasattr(provider_instance, 'model_map'):
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
        if hasattr(provider, 'model_map'):
            models = set(provider.model_map.keys())  # Fetch model names from model_map
            model_sets.append(models)

    common_models = set.intersection(*model_sets) if model_sets else set()
    return list(common_models)

# Validate user-selected models
def validate_selected_models(selected_models, common_models, selected_providers):
    """Validates user-selected models and returns a list of valid models."""    
    valid_models = []
    for model in selected_models:
        if model in common_models:
            valid_models.append(model)
        else:
            if len(selected_providers) > 1:
                print(f"Warning: Model '{model}' is not a common model among the chosen providers. \
                Please select common models.")
            else:
                for provider in selected_providers:
                    if model in provider.model_map:
                        valid_models.append(model)
                        break
                    print(f"Warning: Model '{model}' not available for all selected providers.")
    return valid_models

# Main function to run the benchmark
def run_benchmark(config):
    """Runs the benchmark based on the given configuration."""
    providers = config.get("providers", [])
    num_requests = config.get("num_requests", 1)
    models = config.get("models", [])
    input_tokens = config.get("input_tokens", 10)
    streaming = config.get("streaming", False)
    max_output = config.get("max_output", 100)
    verbose = config.get("verbose", False)
    # Validate and initialize providers
    selected_providers = validate_providers(providers)
    print(f"Selected Providers: {[provider.__class__.__name__ for provider in selected_providers]}")
    
    # Get common models from selected providers
    common_models = get_common_models(selected_providers) if len(selected_providers) > 1 else []
    if not common_models and len(selected_providers) > 1:
        # logging.error("No common models found among selected providers.")
        print("No common models found among selected providers.")
        return

    # Validate models
    valid_models = validate_selected_models(models, common_models, selected_providers)
    if not valid_models:
        print("No valid/common models selected. Ensure models are available across providers.")
        display_available_providers()
        return

    # logging.info(f"Selected Models: {valid_models}")
    print(f"Selected Models: {valid_models}")

    # handling input tokens
    if input_tokens not in input_sizes:
        print(f"Please enter an input token from the following choices: {input_sizes}")
        return
    
    prompt = get_prompt(input_tokens)
    # print(f"Prompt: {prompt}")

    if max_output < OUTPUT_SIZE_LOWER_LIMIT or max_output > OUTPUT_SIZE_UPPER_LIMIT:
        print(f"Please enter an output token length between \
            {OUTPUT_SIZE_LOWER_LIMIT} and {OUTPUT_SIZE_UPPER_LIMIT}.")
        return
    # Run the benchmark
    # logging.info("\nRunning benchmark...")

    print("\nRunning benchmark...")
    benchmark = Benchmark(
        selected_providers, num_requests, valid_models, max_output, 
        prompt=prompt, streaming=streaming, verbosity=verbose
    )
    benchmark.run()

def main():
    """Main function to parse arguments and run the program."""
    args = parser.parse_args()
    # Display available providers and models if --list flag is used
    if args.list:
        display_available_providers()
    elif args.config:
        config = load_config(args.config)
        if config:
            run_benchmark(config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

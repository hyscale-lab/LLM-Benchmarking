"""
Module for benchmarking performance metrics of Large Language Model providers and storing
results in a DynamoDB table.
"""
import json
import os
import time
import uuid
from datetime import datetime
import matplotlib.pyplot as plt
import boto3
import numpy as np
from botocore.exceptions import ClientError
from matplotlib.ticker import LogLocator, FormatStrFormatter

class Benchmark:
    """
    A class to benchmark the performance of various LLM providers.
    """

    def __init__(
        self,
        providers,
        num_requests,
        models,
        max_output,
        prompt,
        streaming=False,
        verbosity=False,
        vllm_ip=None
    ):
        """
        Initialize the Benchmark object.

        Args:
            providers (list): List of provider objects to benchmark.
            num_requests (int): Number of requests to send per provider.
            models (list): List of models to benchmark.
            max_output (int): Maximum output length.
            prompt (str): Input prompt for benchmarking.
            streaming (bool, optional): Whether to use streaming mode. Defaults to False.
            verbosity (bool, optional): Enable verbose output. Defaults to False.
        """
        self.providers = providers
        self.num_requests = num_requests
        self.models = models
        self.prompt = prompt
        self.streaming = streaming
        self.max_output = max_output
        self.verbosity = verbosity
        self.run_id = str(uuid.uuid4())  # Generate a unique ID for each benchmark run

        base_dir = "streaming" if streaming else "end_to_end"

        provider_names = sorted(
            [provider.__class__.__name__.lower() for provider in providers]
        )
        provider_dir_name = "_".join(provider_names)

        self.graph_dir = os.path.join("benchmark_graph", base_dir, provider_dir_name)

        # Create directories if they don't exist
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

        # Initialize DynamoDB
        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        self.table_name = "BenchmarkMetrics"  # Replace with your DynamoDB table name

        # Data structure to hold all metrics for this run
        self.benchmark_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt": self.prompt,
            "providers": {},
        }

    @staticmethod
    def clean_data(data):
        """
        Recursively removes empty values from a dictionary.

        Args:
            data (dict): Input dictionary.

        Returns:
            dict: Cleaned dictionary.
        """
        if isinstance(data, dict):
            return {
                k: Benchmark.clean_data(v)
                for k, v in data.items()
                if v not in [None, "", [], {}]
            }
        if isinstance(data, list):
            return [Benchmark.clean_data(v) for v in data if v not in [None, "", [], {}]]
        return str(data) if isinstance(data, float) else data

    def store_data_points(self):
        """
        Store benchmark data in DynamoDB using an optimized schema with nested metrics.
        """
        table = self.dynamodb.Table(self.table_name)
        try:
            for provider_name, models in self.benchmark_data["providers"].items():
                for model_name, metrics in models.items():
                    model_key = "common" if self.models[0] == "common-model" else "multi"
                    item = {
                        "id": str(uuid.uuid4()),
                        "run_id": self.benchmark_data["run_id"],
                        "timestamp": self.benchmark_data["timestamp"],
                        "provider_name": provider_name,
                        "model_name": model_name,
                        "model_key": model_key,
                        "prompt": self.benchmark_data["prompt"],
                        "metrics": json.dumps(metrics),  # Serialize metrics as JSON string
                        "streaming": self.streaming,
                    }
                    print(item)
                    table.put_item(Item=item)
            print(f"Successfully stored benchmark data for run ID {self.run_id}")
        except ClientError as e:
            print(f"Error saving to DynamoDB: {e.response['Error']['Message']}")

    def add_metric_data(self, provider_name, model_name, metric, latencies):
        """
        Add latency and CDF data to the benchmark data structure.

        Args:
            provider_name (str): The name of the provider.
            model_name (str): The name of the model.
            metric (str): The metric type (e.g., response_times, timetofirsttoken).
            latencies (list): List of latency values in milliseconds.
        """
        latencies_sorted = np.sort(latencies) * 1000  # Convert to milliseconds
        cdf = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)

        # Convert floats to string for DynamoDB compatibility
        latencies_sorted = [str(val) for val in latencies_sorted.tolist()]
        cdf = [str(val) for val in cdf.tolist()]

        self.benchmark_data["providers"].setdefault(provider_name, {}).setdefault(
            model_name, {}
        )[metric] = {"latencies": latencies_sorted, "cdf": cdf}

    def plot_metrics(self, metric):
        """
        Plots and saves graphs for the given metric.

        Args:
            metric (str): The name of the metric to plot (e.g., "response_times").
            filename_suffix (str): Suffix to append to the filename for saving the plot.
        """
        plt.figure(figsize=(8, 8))

        for provider in self.providers:
            provider_name = provider.__class__.__name__
            for model, latencies in provider.metrics[metric].items():
                model_name = provider.get_model_name(model)
                self.add_metric_data(provider_name, model_name, metric, latencies)
                # Convert to milliseconds and sort for CDF
                latencies_sorted = np.sort(latencies) * 1000
                cdf = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)
                model_name = provider.get_model_name(model)

                if provider_name.lower() == "vllm":
                    plt.plot(
                        latencies_sorted,
                        cdf,
                        marker="o",
                        linestyle="-",
                        markersize=6,  # Slightly larger marker size
                        color="black",  # Black color for the marker
                        label=f"{provider_name} - {model_name}",
                        linewidth=2,  # Bold line
                    )
                else:
                    plt.plot(
                        latencies_sorted,
                        cdf,
                        marker="o",
                        linestyle="-",
                        markersize=5,
                        label=f"{provider_name} - {model_name}",
                    )
                
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Portion of requests", fontsize=12)
        plt.grid(True)

        # Add legend
        plt.legend(loc="best")
        plt.xscale("log")
        # **Ensure all ticks are labeled**
        ax = plt.gca()

        # display 5 minor ticks between each major tick
        # minorLocator = LogLocator(subs=np.linspace(2, 10, 6, endpoint=False))
        minorLocator = LogLocator(base=10.0, subs='auto')
        # format the labels (if they're the x values)
        minorFormatter = FormatStrFormatter('%1.1f')
        
        # for no labels use default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        
        ax.xaxis.set_minor_formatter(minorFormatter)
        for label in ax.get_xminorticklabels():
            label.set_fontsize(8)   # smaller font for minor labels
            label.set_rotation(45)  # rotate 90 degrees for readability
        plt.tight_layout()

        current_time = datetime.now().strftime("%y%m%d_%H%M")
        filename = f"{metric}_{current_time}.png"
        filepath = os.path.join(self.graph_dir, filename)
        plt.savefig(filepath)
        plt.close()

        print(f"Saved graph: {filepath}")

    def run(self):
        """
        Execute the benchmark and store metrics in DynamoDB.
        """
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            print(f"{provider_name}")
            for model in self.models:
                for i in range(self.num_requests):
                    if self.verbosity:
                        print(f"Request {i + 1}/{self.num_requests}")

                    if i % 10 == 0:
                        # print("[DEBUG] Sleeping for 2 mins to bypass rate limit...")
                        time.sleep(120)
                        print("finished sleeping")

                    if self.streaming:
                        if provider_name == "vLLM":
                            provider.perform_inference_streaming(
                                model, self.prompt, self.vllm_ip, self.max_output, self.verbosity
                            )
                        else:
                            provider.perform_inference_streaming(
                                model, self.prompt, self.max_output, self.verbosity
                            )
                    else:
                        if provider_name == "vLLM":
                            provider.perform_inference(
                                model, self.prompt, self.vllm_ip, self.max_output, self.verbosity
                            )
                        else:
                            provider.perform_inference(
                                model, self.prompt, self.max_output, self.verbosity
                            )

        metrics_to_plot = (
            ["timetofirsttoken", "response_times", "timebetweentokens", "tps", "timebetweentokens_p95", "timebetweentokens_median"]
            if self.streaming
            else ["response_times"]
        )
        
        for metric in metrics_to_plot:
            self.plot_metrics(metric)
            
        self.store_data_points()

        

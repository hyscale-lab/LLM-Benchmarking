import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime
from matplotlib.ticker import LogLocator, FormatStrFormatter
from trace.proxy import ProxyServer
from trace.loadgenerator import LoadGenerator

class Benchmark:
    """
    A class to run and visualize benchmarks for different AI providers.

    Attributes:
        providers (list): List of AI provider instances.
        num_requests (int): Number of requests to run per model.
        models (list): List of model names to benchmark.
        max_output (int): Maximum number of tokens for model output.
        prompt (str): The input prompt to use for benchmarking.
        streaming (bool): Flag to indicate whether to use streaming mode.
        verbosity (bool): Flag to enable verbose output during benchmarking.
        graph_dir (str): Directory path for saving generated plots.
    """

    def __init__(
        self,
        providers,
        num_requests,
        models,
        max_output,
        prompt,
        streaming=False,
        input_type="static",
        verbosity=False,
        vllm_ip=None,
        dataset=None
    ):
        """
        Initializes the Benchmark instance with provided parameters.

        Args:
            providers (list): List of AI provider instances.
            num_requests (int): Number of requests to run per model.
            models (list): List of model names to benchmark.
            max_output (int): Maximum number of tokens for model output.
            prompt (str): The input prompt to use for benchmarking.
            streaming (bool, optional): Flag to indicate streaming mode. Defaults to False.
            verbosity (bool, optional): Flag to enable verbose output. Defaults to False.
        """
        self.providers = providers
        self.num_requests = num_requests
        self.models = models
        self.prompt = prompt
        self.streaming = streaming
        self.input_type = input_type
        self.max_output = max_output
        self.verbosity = verbosity
        self.vllm_ip = vllm_ip
        self.dataset = dataset

        base_dir = "streaming" if streaming else "end_to_end"

        provider_names = sorted(
            [provider.__class__.__name__.lower() for provider in providers]
        )
        provider_dir_name = "_".join(provider_names)

        if self.input_type == "static":
            self.graph_dir = os.path.join("benchmark_graph", base_dir, provider_dir_name)
        elif self.input_type == "trace":
            self.graph_dir = os.path.join("benchmark_graph", "trace", base_dir, provider_dir_name)
        elif self.input_type == "multiturn":
            self.graph_dir = os.path.join("benchmark_graph", "multiturn", base_dir, provider_dir_name)

        # Create directories if they don't exist
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

    def plot_metrics(self, metric, filename_suffix):
        """
        Plots and saves graphs for the given metric.

        Args:
            metric (str): The name of the metric to plot (e.g., "response_times").
            filename_suffix (str): Suffix to append to the filename for saving the plot.
        """
        plt.figure(figsize=(8, 8))
        any_series = False

        for provider in self.providers:
            provider_name = provider.__class__.__name__
            for model, latencies in provider.metrics[metric].items():
                if len(latencies) == 0:
                    continue
                # Convert to milliseconds and sort for CDF
                latencies_sorted = np.sort(latencies) * 1000
                cdf = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)
                model_name = provider.get_model_name(model)

                median = float(np.percentile(latencies, 50))
                p95 = float(np.percentile(latencies, 95))
                mean = float(np.mean(latencies))

                if self.verbosity:
                    print(f"Results for {provider_name}({model}) - {metric}:")
                    print(f"Median {metric}: {median}")
                    print(f"P95 {metric}: {p95}")
                    print(f"Mean {metric}: {mean}")

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
                any_series = True
                
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Portion of requests", fontsize=12)
        plt.grid(True)

        # Add legend
        if any_series:
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
        filename = f"{filename_suffix}_{current_time}.png"
        filepath = os.path.join(self.graph_dir, filename)
        plt.savefig(filepath)
        plt.close()

        print(f"Saved graph: {filepath}")

    def run(self):
        """
        Runs the benchmark for the selected providers and models, and plots the results.

        This method sends a number of requests to each model for each provider, collects
        performance metrics, and generates plots based on those metrics.
        """
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            print(f"{provider_name}")
            for model in self.models:
                model_name = provider.get_model_name(model)
                print(f"Model: {model_name}\nPrompt: {self.prompt}")

                # --- START TIME ---
                start_time = datetime.now()
                print(f"Start Time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                # ------------------

                for i in range(self.num_requests):
                    if self.verbosity:
                        print(f"Request {i + 1}/{self.num_requests}")

                    if i % 20 == 0:
                        time.sleep(120)

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

                # --- FINISH TIME & DURATION ---
                end_time = datetime.now()
                duration = end_time - start_time
                print(f"Finish Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Duration:    {duration}")
                # ------------------------------

        if not self.streaming:
            self.plot_metrics("response_times", "response_times")
        else:
            # Save all the relevant metrics plots when streaming is true
            self.plot_metrics("timetofirsttoken", "timetofirsttoken")
            self.plot_metrics("response_times", "totaltime")
            self.plot_metrics("timebetweentokens", "timebetweentokens")

        if self.dataset:
            print("\nRunning accuracy evaluation...")
            for provider in self.providers:
                model_names = provider.get_model_name("reasoning-model")
                if not model_names:
                    continue
                for model_name in model_names:
                    if not hasattr(provider, "measure_accuracy"):
                        print(f"[SKIP] {provider.__class__.__name__} has no measure_accuracy().")
                        continue
                    try:
                        acc_summary = provider.measure_accuracy(
                            dataset=self.dataset,
                            verbosity=self.verbosity,
                            metric_default="numeric",
                            model_id=model_name,
                        )
                        provider.log_metrics(
                            model_name, "aime_2024_accuracy", acc_summary.get("accuracy")
                        )
                        print(f"[ACCURACY] {provider.__class__.__name__} – "f"{model_name}: {acc_summary}")
                    except Exception as e:
                        print(f"[WARN] measure_accuracy failed for "f"{provider.__class__.__name__}/{model_name}: {e!r}")

    def run_trace(self):
        """
        Runs the benchmark for the selected providers and models, and plots the results.

        This method sends a number of requests to each model for each provider, collects
        performance metrics, and generates plots based on those metrics.
        """
        for provider in self.providers:
            provider_name = provider.__class__.__name__

            print(f"Starting proxy server for {provider_name}...")
            proxy_server = ProxyServer()
            proxy_server.start()
            while not getattr(proxy_server.server, 'started', False):  # Wait for server startup
                pass
            print("Loading load generator...")
            load_generator = LoadGenerator(proxy_server.get_url())

            for model in self.models:
                model_name = provider.get_model_name(model)
                print(f"\n[{provider_name}] - Model: {model_name}")

                # --- START TIME ---
                start_time = datetime.now()
                print(f"Start Time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                # ------------------

                if provider_name == "vLLM":
                    provider.perform_trace(model, proxy_server, load_generator, self.streaming, self.num_requests, self.verbosity, self.vllm_ip)
                else:
                    provider.perform_trace(model, proxy_server, load_generator, self.streaming, self.num_requests, self.verbosity)

                # --- FINISH TIME & DURATION ---
                end_time = datetime.now()
                duration = end_time - start_time
                print(f"Finish Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Duration:    {duration}")
                # ------------------------------

            print(f"Stopping proxy server for {provider_name}...")
            proxy_server.stop()
        
        print()

        if not self.streaming:
            self.plot_metrics("response_times", "response_times")
        else:
            # Save all the relevant metrics plots when streaming is true
            self.plot_metrics("timetofirsttoken", "timetofirsttoken")
            self.plot_metrics("response_times", "totaltime")
            self.plot_metrics("timebetweentokens", "timebetweentokens")

    def run_multiturn(self, time_interval):
        """
        Runs the benchmark for the selected providers and models, and plots the results.

        This method sends a number of requests to each model for each provider, collects
        performance metrics, and generates plots based on those metrics.
        """
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            for model in self.models:
                model_name = provider.get_model_name(model)
                print(f"\n[{provider_name}] - Model: {model_name}")

                # --- START TIME ---
                start_time = datetime.now()
                print(f"Start Time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                # ------------------

                if provider_name == "vLLM":
                    provider.perform_multiturn(model, time_interval, self.streaming, self.num_requests, self.verbosity, self.vllm_ip)
                else:
                    provider.perform_multiturn(model, time_interval, self.streaming, self.num_requests, self.verbosity)

                # --- FINISH TIME & DURATION ---
                end_time = datetime.now()
                duration = end_time - start_time
                print(f"Finish Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Duration:    {duration}")
                # ------------------------------

        print()

        if not self.streaming:
            self.plot_metrics("response_times", "response_times")
        else:
            # Save all the relevant metrics plots when streaming is true
            self.plot_metrics("timetofirsttoken", "timetofirsttoken")
            self.plot_metrics("response_times", "totaltime")
            self.plot_metrics("timebetweentokens", "timebetweentokens")

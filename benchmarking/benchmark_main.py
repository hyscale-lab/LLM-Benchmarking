import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime
from matplotlib.ticker import LogLocator, FormatStrFormatter
import random
from utils.db_utils import save_flattened_metrics_to_csv
from utils.prompt_generator import generate_prompt

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
        outputs,
        # prompts,
        inputs,
        dataset,
        streaming=False,
        verbosity=False,
        vllm_ip=None,
        exp_dir=None
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
        # self.prompts = prompts
        self.inputs = inputs
        self.dataset = dataset
        self.streaming = streaming
        self.outputs = outputs
        self.verbosity = verbosity
        self.vllm_ip = vllm_ip
        self.exp_dir = exp_dir
        # New parameters for retry mechanism
        self.max_retries = 3
        self.base_timeout = 10

        self.prompts = {}
        self.answers = {}
        print("PROMPTS")

        for input_size in self.inputs:
            self.prompts[input_size] = []
            self.answers[input_size] = []
            for n in range(self.num_requests):
                # print(n, input_size)
                if dataset == "aime":
                    print(n)
                    prompt, correct_answer = generate_prompt(dataset, n, input_size)
                else:
                    prompt = generate_prompt(dataset, n, input_size)
                self.prompts[input_size].append(prompt)
                if self.dataset == "aime":
                    self.answers[input_size].append(correct_answer)
                
                print(prompt[:200])
                # print(correct_answer[:200])
                # print(len(prompt.split()))
                # print(len(self.prompts[input_size]))

        base_dir = "streaming" if streaming else "end_to_end"

        provider_names = sorted(
            [provider.__class__.__name__.lower() for provider in providers]
        )
        provider_dir_name = "_".join(provider_names)

        self.graph_dir = os.path.join("benchmark_graph", base_dir, provider_dir_name)

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

        for provider in self.providers:
            provider_name = provider.__class__.__name__
            print(metric)
            save_flattened_metrics_to_csv(provider, metric, f"{self.exp_dir}/{metric}_logs.csv")
            if metric == "dpsk_output" or metric == "extracted_answer" or metric == "correct_answer":
                continue
            # print(provider.metrics[metric].items())
            all_lats = []
            for model, input_dict in provider.metrics[metric].items():
                for input_size, output_dict in input_dict.items():
                    for max_output, latencies in output_dict.items():
                        # Convert to milliseconds and sort for CDF
                        print(model, input_size, len(latencies))
                        if metric == "timebetweentokens":
                            for sub in latencies:
                                all_lats.extend(sub)
                        else:
                            all_lats.extend(latencies)
                        
            latencies_sorted = np.sort(np.array(all_lats)) * 1000
            cdf = np.arange(1, len(latencies_sorted) + 1) / len(latencies_sorted)
                    # model_name = provider.get_model_name(model)
            if provider_name.lower() == "vllm":
                plt.plot(
                    latencies_sorted,
                    cdf,
                    marker="o",
                    linestyle="-",
                    markersize=6,  # Slightly larger marker size
                    color="black",  # Black color for the marker
                    # label=f"{provider_name} - {model_name}",
                    label=f"{provider_name}",
                    linewidth=2,  # Bold line
                )
            else:
                plt.plot(
                    latencies_sorted,
                    cdf,
                    marker="o",
                    linestyle="-",
                    markersize=5,
                    # label=f"{provider_name} - {model_name}",
                    label=f"{provider_name}",
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
        print("running here!!!!")
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            # logging.debug(f"{provider_name}")
            # print(f"{provider_name}")
            
            for model in self.models:
                model_name = provider.get_model_name(model)
                # print(f"Model: {model_name}")
                
                for input_size in self.inputs:
                    print(f"Prompt size: {input_size}")
                    
                    for max_output in self.outputs:
                        
                        for i in range(self.num_requests):
                            if self.verbosity:
                                print(f"Request {i + 1}/{self.num_requests}")
                                print(f"{provider_name}" + f" Model: {model_name}")

                            if provider_name == "AWSBedrock" and ((i+1) % 10) == 0:
                                print("[DEBUG] Sleeping for 10s to bypass rate limit...")
                                time.sleep(10)
                            
                            if provider_name == "TogetherAI" and self.dataset == 'aime':
                                time.sleep(60)

                            #     print("[DEBUG] Finished.")
                            # if ((i+1) % 58) == 0:
                            #     time.sleep(120)
                            # prompt = get_prompt(input_size)
                            # print(prompt)

                            #     prompt = "Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."
                            # else:
                            #     prompt = self.prompts[input_size][i]
                            correct_answer = None
                            prompt = self.prompts[input_size][i]
                            if self.dataset == "aime":
                                correct_answer = self.answers[input_size][i]
                                print(prompt[:200])
                                print(correct_answer[:200])

                            # print(f"{prompt[:200]}")
                            if self.streaming:
                                if provider_name == "vLLM":
                                    provider.perform_inference_streaming(
                                        model, prompt, self.vllm_ip, max_output, self.verbosity, correct_answer
                                    )
                                else:
                                    provider.perform_inference_streaming(
                                        model, prompt, max_output, self.verbosity, correct_answer
                                    )
                            else:
                                if provider_name == "vLLM":
                                    provider.perform_inference(
                                        model, prompt, self.vllm_ip, max_output, self.verbosity
                                    )
                                else:
                                    provider.perform_inference(
                                        model, prompt, max_output, self.verbosity
                                    )


                            # # Simple retry mechanism
                            
                            # success = False
                            # attempts = 0
                            
                            # while not success and attempts < self.max_retries:
                            #     try:
                            #         # Calculate timeout with exponential backoff
                            #         timeout = self.base_timeout * (2 ** attempts)
                            #         print(timeout)
                                    
                            #         if self.streaming:
                            #             if provider_name == "vLLM":
                            #                 provider.perform_inference_streaming(
                            #                     model, self.prompt, self.vllm_ip, 
                            #                     self.max_output, self.verbosity, timeout=timeout
                            #                 )
                            #             else:
                            #                 provider.perform_inference_streaming(
                            #                     model, self.prompt, timeout, self.max_output, 
                            #                     self.verbosity
                            #                 )
                            #         else:
                            #             if provider_name == "vLLM":
                            #                 provider.perform_inference(
                            #                     model, self.prompt, self.vllm_ip, 
                            #                     self.max_output, self.verbosity, timeout=timeout
                            #                 )
                            #             else:
                            #                 provider.perform_inference(
                            #                     model, self.prompt, self.max_output, 
                            #                     self.verbosity, timeout=timeout
                            #                 )
                                    
                            #         # If we get here, the request was successful
                            #         success = True
                                    
                            #     except Exception as e:
                            #         attempts += 1
                            #         if self.verbosity:
                            #             print(f"Request failed (attempt {attempts}/{self.max_retries}): {str(e)}")
                                    
                            #         # Add some jitter to avoid request storms on retry
                            #         if attempts < self.max_retries:
                            #             jitter = random.uniform(0.5, 1.5)
                            #             wait_time = min(30, (2 ** attempts) * jitter)
                            #             time.sleep(wait_time)
                            
                            # # Record failed request if all attempts failed
                            # if not success:
                            #     print(f"Request {i+1} failed after {self.max_retries} attempts")
        if not self.streaming:
            self.plot_metrics("response_times", "response_times")
        else:
            # Save all the relevant metrics plots when streaming is true
            self.plot_metrics("timetofirsttoken", "timetofirsttoken")
            self.plot_metrics("response_times", "totaltime")
            self.plot_metrics("timebetweentokens", "timebetweentokens")
            self.plot_metrics("timebetweentokens_median", "timebetweentokens_median")
            self.plot_metrics("timebetweentokens_p95", "timebetweentokens_p95")
            self.plot_metrics("timebetweentokens_p99", "timebetweentokens_p99")
            self.plot_metrics("timebetweentokens_avg", "timebetweentokens_avg")
            self.plot_metrics("totaltokens", "totaltokens")
            self.plot_metrics("accuracy", "accuracy")
            self.plot_metrics("dpsk_output", "dpsk_output")
            self.plot_metrics("extracted_answer","extracted_answer")
            self.plot_metrics("correct_answer", "correct_answer")

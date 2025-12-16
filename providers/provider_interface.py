import os
from abc import ABC, abstractmethod


# create an interface for providers (abstract class)
class ProviderInterface(ABC):

    def __init__(self):
        """
        Initializes the Provider with the necessary API key and client.
        """
        # experiment constants
        self.min_tokens = 10000
        self.system_prompt = (
            f"Please provide a detailed response of MORE THAN {self.min_tokens} words"
        )

        # metrics
        self.metrics = {
            "response_times": {},
            "response_times_median": {},
            "response_times_p95": {},
            "timetofirsttoken": {},
            "timetofirsttoken_median": {},
            "timetofirsttoken_p95": {},
            "totaltokens": {},
            "tps": {},
            "tps_median": {},
            "tps_p95": {},
            "timebetweentokens": {},
            "timebetweentokens_median": {},
            "timebetweentokens_p95": {},
            "aime_2024_accuracy": {}
        }

        # for trace input type
        self.trace_dataset_path = os.getenv('TRACE_DATASET_PATH', './trace/sample.json')
        self.trace_result_path = f'./trace/{self.__class__.__name__}.result'

    def log_metrics(self, model_name, metric, value):
        """
        Logs metrics
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric type '{metric}' is not defined.")
        if model_name not in self.metrics[metric]:
            self.metrics[metric][model_name] = []

        self.metrics[metric][model_name].append(value)

    @abstractmethod
    def perform_inference(self, model, prompt):
        """
        perform_inference
        """

    @abstractmethod
    def perform_inference_streaming(self, model, prompt):
        """
        perform_inference_streaming
        """

    @abstractmethod
    def perform_trace(self, proxy_server, load_generator, num_requests, streaming, verbosity, model):
        """
        perform_trace
        """

    @abstractmethod
    def get_model_name(self, model):
        """
        get model names
        """

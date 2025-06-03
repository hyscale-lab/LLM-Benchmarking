from abc import ABC, abstractmethod


# create an interface for providers (abstract class)
class ProviderInterface(ABC):

    def __init__(self):
        """
        Initializes the Provider with the necessary API key and client.
        """
        # experiment constants
        self.min_tokens = 100000
        self.system_prompt = (
            f"Please provide a detailed response of STRICTLY MORE THAN {self.min_tokens} words"
        )

        # metrics
        self.metrics = {
            "response_times": {},
            "timetofirsttoken": {},
            "totaltokens": {},
            "tps": {},
            "timebetweentokens": {},
            "timebetweentokens_median": {},
            "timebetweentokens_p95": {},
        }

    # def log_metrics(self, model_name, input_size, metric, value):
    #     """
    #     Logs metrics
    #     """
    #     if metric not in self.metrics:
    #         raise ValueError(f"Metric type '{metric}' is not defined.")
    #     if model_name not in self.metrics[metric]:
    #         self.metrics[metric][model_name] = []

    #     self.metrics[metric][model_name].append(value)

    def log_metrics(self, model_name, input_size, max_output, metric, value):
        """
        Logs metrics in a nested structure: metrics[metric][model_name][input_size] -> list of values
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric type '{metric}' is not defined.")
        
        if model_name not in self.metrics[metric]:
            self.metrics[metric][model_name] = {}

        if input_size not in self.metrics[metric][model_name]:
            self.metrics[metric][model_name][input_size] = {}

        if max_output not in self.metrics[metric][model_name][input_size]:
            self.metrics[metric][model_name][input_size][max_output] = []

        self.metrics[metric][model_name][input_size][max_output].append(value)

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
    def get_model_name(self, model):
        """
        get model names
        """

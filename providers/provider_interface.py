import os
import asyncio
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
    def get_model_name(self, model):
        """
        get model names
        """

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

    def perform_trace(self, proxy_server, load_generator, num_requests, streaming, verbosity, model='common-model'):
        """
        Perform using trace input
        """
        # Set handler for proxy
        async def data_handler(data):
            prompt = data.pop('prompt')
            gen_tokens = data.pop('generated_tokens')

            def inference_sync():
                try:
                    if streaming:
                        response_list = self.perform_inference_streaming(model, prompt, gen_tokens, verbosity)
                        if isinstance(response_list, Exception):
                            return [{"error": f"Inference failed: {response_list}"}]
                        return response_list
                    else:
                        response = self.perform_inference(model, prompt, gen_tokens, verbosity)
                        if isinstance(response, Exception):
                            return {"error": f"Inference failed: {response}"}
                        return response

                except Exception as e:
                    print(f"\nData handling failed: {e}")
                    return [{"error": f"Data handling failed: {e}"}] if streaming else {"error": f"Data handling failed: {e}"}

            response = await asyncio.to_thread(inference_sync)
            return response

        proxy_server.set_handler(data_handler)
        proxy_server.set_streaming(streaming)

        # Start load generator
        load_generator.send_loads(
            self.trace_dataset_path,
            self.trace_result_path,
            sampling_rate=50,
            recur_step=10,
            limit=num_requests,
            max_drift=10000,
            upscale='ars'
        )

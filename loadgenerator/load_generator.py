"""
Module for running a custom load generator binary against a target URL.

This wraps subprocess execution of the LLMLoadgen binary with parameters
"""

import subprocess
from pathlib import Path

class LoadGenerator():
    """
    Wrapper class for executing the LLMLoadgen binary with configurable parameters.
    """

    def __init__(self, target_url):
        """
        Initialize a LoadGenerator.

        Args:
            target_url (str): The target URL to send load against.
        """

        self.binary = './loadgenerator/LLMLoadgen'
        self.target_url = target_url

    def send_loads(
        self, dataset_path, result_path, sampling_rate, recur_step, limit, max_drift, upscale=None
    ):
        """
        Run the load generator with the specified parameters.
        """

        # check dataset_path exists
        dataset_abs_path = Path(dataset_path).resolve()
        if not dataset_abs_path.exists:
            raise FileNotFoundError(f'{dataset_path} (abs path: {dataset_abs_path})')

        # generate command for load generator
        cmd = [
            self.binary,
            "-url", self.target_url,
            "-dataset_path", str(dataset_abs_path),
            "-result_path", result_path,
            "-sampling_rate", str(sampling_rate),
            "-recur_step", str(recur_step),
            "-limit", str(limit),
            "-max_drift", str(max_drift)
        ]
        if upscale:
            cmd.extend(["-upscale", upscale])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            # print(f'LoadGenerator result: {result}')
        except subprocess.CalledProcessError as e:
            print(f"LoadGenerator failed: {e}, stderr: {e.stderr}")

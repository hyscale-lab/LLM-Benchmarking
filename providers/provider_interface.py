from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import re

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
            "accuracy": {}
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
        # print(model_name, input_size, max_output, metric, value)

        if metric not in self.metrics:
            raise ValueError(f"Metric type '{metric}' is not defined.")
        
        if model_name not in self.metrics[metric]:
            self.metrics[metric][model_name] = {}

        if input_size not in self.metrics[metric][model_name]:
            self.metrics[metric][model_name][input_size] = {}

        if max_output not in self.metrics[metric][model_name][input_size]:
            self.metrics[metric][model_name][input_size][max_output] = []

        self.metrics[metric][model_name][input_size][max_output].append(value)
    
    
    def extract_answer_aime(self, content: Optional[str]) -> Optional[str]:
        """Extracts the final three-digit answer from the model's response.
        
        Args:
            content (Optional[str]): The model's response text
            
        Returns:
            Optional[str]: The extracted answer as a three-digit string (e.g., '042'),
                        or None if no valid answer is found
        """
        if content is None:
            logger.error("[red]AIME extract_answer: Content is None.[/red]")
            return None

        # Regex to find \boxed{NUMBER} pattern, accepting any number
        # In a string literal, \\boxed means \boxed in the actual content
        BOXED_PATTERN = r'\\boxed\{(\d+)\}'

        # Search the entire content, but prioritize the last match if multiple exist
        matches = list(re.finditer(BOXED_PATTERN, content))
        
        # Alternative regex patterns to try if the first one doesn't work
        if not matches and content:
            # Try with different patterns for robustness
            alt_patterns = [
                r'boxed\{(\d+)\}',        # In case the backslash is missing
                r'\\boxed\{\s*(\d+)\s*\}', # Allow spaces inside braces
                r'boxed\{\s*(\d+)\s*\}'    # Missing backslash + spaces
            ]
            
            for pattern in alt_patterns:
                matches = list(re.finditer(pattern, content))
                if matches:
                    break

        if not matches:
            logger.error(f"[red]No \\boxed{{}} answer found in content.[/red]")
            return None

        # Get the last match found
        last_match = matches[-1]
        extracted_number = last_match.group(1)
        
        # Convert to integer and format as string
        try:
            num = int(extracted_number)
            return str(num)
        except ValueError:
            logger.error(f"[red]Invalid number format in AIME answer: {extracted_number}.[/red]")
            return None

    def calculate_score_aime(self, extracted_answer: Optional[str], correct_answer: str) -> int:
        """Calculates the score for an AIME problem (1 for correct, 0 for incorrect/missing).
        
        Args:
            extracted_answer (Optional[str]): The extracted answer from the model's response
            correct_answer (str): The correct answer for the problem
            
        Returns:
            int: 1 if the answers match exactly, 0 otherwise
        """
        # Handle None or invalid extracted answer
        if extracted_answer is None:
            logger.error(f"[red]No valid AIME answer found in extracted_answer.[/red]")
            return 0
            
        try:
            # Convert both to integers for comparison (strips leading zeros)
            extracted_num = int(extracted_answer)
            correct_num = int(correct_answer)

            # Compare the numbers
            return 1 if extracted_num == correct_num else 0
            
        except ValueError:
            # Handle case where either string isn't a valid integer
            logger.error(f"[red]Invalid AIME numbers: extracted={extracted_answer}, correct={correct_answer}.[/red]")
            return 0 


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
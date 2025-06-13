# logger_utils.py
import logging
import os
from datetime import datetime

def get_shared_logger() -> logging.Logger:
    log_dir = "inference_logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"benchmark_{timestamp}.log")

    logger = logging.getLogger("LLMbenchmark")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

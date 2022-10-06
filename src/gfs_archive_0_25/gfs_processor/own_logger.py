import logging
import sys
import os
from pathlib import Path


def get_logger(log_path: str = os.path.join(Path(__file__).parent, "log/gfs_processor.log")):
    logger = logging.getLogger(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger

logger = get_logger()
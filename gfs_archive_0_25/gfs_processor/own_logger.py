import logging
import sys
import os

def get_logger():
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    if not os.path.exists('log'):
        os.mkdir('log')

    handler = logging.FileHandler('log/gfs_processor.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger

logger = get_logger()
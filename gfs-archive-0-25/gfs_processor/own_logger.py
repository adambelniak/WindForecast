import logging
import sys

def get_logger():
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handler = logging.FileHandler('gfs_prcessor.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger

logger = get_logger()
from __future__ import annotations

import os

import coolname  # type: ignore
from dotenv import load_dotenv

load_dotenv()


def setup_rundir():
    """
    Create a working directory with a randomly generated run name.
    """
    if os.getenv('RUN_NAME') is None:
        name = coolname.generate_slug(2)  # type: ignore
        os.environ['RUN_NAME'] = f'{name}'

    results_root = f'{os.getenv("RESULTS_DIR")}/{os.getenv("WANDB_PROJECT")}'
    if os.getenv('RUN_MODE', '').lower() in ['debug', 'tune_debug', 'analysis']:
        run_dir = f'{results_root}/_{os.getenv("RUN_MODE", "")}/{os.getenv("RUN_NAME")}'
        if os.getenv('RUN_MODE', '').lower() in ['tune', 'tune_debug']:
            os.environ['WANDB_MODE'] = 'disabled'
    else:
        run_dir = f'{results_root}/{os.getenv("RUN_NAME")}'

    os.makedirs(run_dir, exist_ok=True)
    os.environ['RUN_DIR'] = run_dir

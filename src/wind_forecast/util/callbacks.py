from __future__ import annotations

import errno
import os
import re
from pathlib import Path
from typing import Any, Optional, Dict
from sys import platform

from pytorch_lightning.utilities.types import _METRIC
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from wandb.sdk.wandb_run import Run


# -------------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------------
from wind_forecast.config.register import Config


class CustomCheckpointer(ModelCheckpoint):
    """
    Default pl.callback.ModelCheckpoint with fixed filenames (_ instead of =).
    """
    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        metrics: Dict[str, _METRIC],
        prefix: str = "",
        auto_insert_metric_name: bool = True
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            for group in groups:
                name = group[1:]
                filename = filename.replace(group, name + "_{" + name)
                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename


def get_resume_checkpoint(cfg: Config, wandb_logger: WandbLogger) -> Optional[str]:
    run: Run = wandb_logger.experiment  # type: ignore
    path = cfg.experiment.resume_checkpoint

    if path is None:
        return None

    wandb_prefix = 'wandb://'

    if path.startswith(wandb_prefix):
        run_name = os.getenv("RUN_NAME")
        # Resuming from wandb artifacts, e.g.:
        # resume_checkpoint: wandb://WANDB_LOGIN/WANDB_PROJECT_NAME/ARTIFACT_NAME:v0@checkpoint.ckpt
        artifact_root = f'{os.getenv("RESULTS_DIR")}/{os.getenv("WANDB_PROJECT")}/{run_name}/artifacts'

        path = path[len(wandb_prefix):]
        artifact_path, checkpoint_path = path.split('@')
        artifact_name = artifact_path.split('/')[-1]
        if platform == "win32":
            artifact_name = artifact_name.replace(':', '-')


        os.makedirs(f'{artifact_root}/{artifact_name}', exist_ok=True)

        artifact = run.use_artifact(artifact_path, type='model')  # type: ignore
        artifact.download()  # type: ignore

        path = f'artifacts/{artifact_name}/{checkpoint_path}'

    if not Path(path).exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

    return path

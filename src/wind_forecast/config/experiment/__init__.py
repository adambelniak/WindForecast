from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf.omegaconf import MISSING


# Experiment settings validation schema & default values
from wind_forecast.preprocess.synop.consts import LSTM_FEATURES


@dataclass
class ExperimentSettings:
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    # wandb tags
    _tags_: Optional[List[str]] = None

    # Seed for all random number generators
    seed: int = 1

    # Path to resume from. Two formats are supported:
    # - local checkpoints: path to checkpoint relative from run (results) directory
    # - wandb artifacts: wandb://ARTIFACT_PATH/ARTIFACT_NAME:VERSION@CHECKPOINT_NAME
    resume_checkpoint: Optional[str] = None

    # Enable checkpoint saving
    save_checkpoints: bool = True

    # Enable initial validation before training
    validate_before_training: bool = True

    # ----------------------------------------------------------------------------------------------
    # Data loading settings
    # ----------------------------------------------------------------------------------------------
    # Training batch size
    batch_size: int = 128

    # Enable dataset shuffling
    shuffle: bool = True

    # Number of dataloader workers
    num_workers: int = 8

    # Model to use
    model: Any = MISSING

    system: Any = MISSING

    # ----------------------------------------------------------------------------------------------
    # Dataset specific settings
    # ----------------------------------------------------------------------------------------------
    # Lightning DataModule to use
    datamodule: Any = MISSING

    val_split: float = .2

    gfs_dataset_dir: str = os.path.join("D:\\WindForecast", "output_np2")

    synop_file: str = "KOZIENICE_488_data.csv"

    target_parameter: str = "temperature"

    prediction_offset: int = 3

    data_dim_x: int = 53

    data_dim_y: int = 33

    train_parameters_config_file: str = "CNNConfig.json"

    input_size: Any = (16, 33, 53)

    epochs: int = 100

    lstm_train_parameters: List = field(default_factory=lambda: LSTM_FEATURES)

    sequence_length: int = 24

from typing import Dict
import torch
from einops import repeat
from pytorch_lightning import LightningModule

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys


class Naive(LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_past_y = batch[BatchKeys.SYNOP_PAST_Y.value].float()
        if self.config.experiment.batch_size == 1:
            last_synop_y = synop_past_y[-1]
            return repeat(last_synop_y, f"len -> (len {self.config.experiment.future_sequence_length})")
        else:
            last_synop_y = synop_past_y[:, -1:]
            return repeat(last_synop_y, f"batch len -> batch (len {self.config.experiment.future_sequence_length})")

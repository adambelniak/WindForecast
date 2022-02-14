from typing import Dict

import torch
from pytorch_forecasting import NBeats
from pytorch_lightning import LightningModule

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys


class NBEATS(LightningModule):

    def __init__(self, config: Config) -> None:
        super(NBEATS, self).__init__()
        self.model = NBeats(stack_types=['generic'],
                            num_blocks=[1 for _ in range(30)],
                            prediction_length=config.experiment.future_sequence_length,
                            context_length=config.experiment.sequence_length,
                            output_transformer=lambda out: out["prediction"])

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        inputs = batch[BatchKeys.SYNOP_PAST_TARGETS.value]
        x = {
            "encoder_cont": inputs.unsqueeze(-1).float(),
            "target_scale": torch.Tensor([[0, 1] for i in range(inputs.size()[0])]).float()
        }

        return self.model(x)['prediction']

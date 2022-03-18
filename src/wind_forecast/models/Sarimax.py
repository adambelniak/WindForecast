from typing import Dict

import pytorch_lightning as pl
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys


class Sarimax(pl.LightningModule):
    def __init__(self,  config: Config):
        super(Sarimax, self).__init__()
        self.config = config

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        exog_input = batch[BatchKeys.GFS_PAST_X.value].float()
        exog_future = batch[BatchKeys.GFS_FUTURE_X.value].float()
        synop_past_observed = batch[BatchKeys.SYNOP_PAST_Y.value].float()

        sarima_model = SARIMAX(synop_past_observed[0].numpy(), exog_input[0].numpy(), (0,1,1), (0, 1, 1, 24))
        model_fit = sarima_model.fit()

        return torch.Tensor(model_fit.forecast(steps=self.config.experiment.future_sequence_length,
                                               exog=exog_future[0].numpy())).unsqueeze(0)

from typing import Dict

import pytorch_lightning as pl
import torch
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys


class Arimax(pl.LightningModule):
    def __init__(self,  config: Config):
        super(Arimax, self).__init__()
        self.config = config
        exp = self.config.experiment
        self.p, self.d, self.q = exp.arima_p, exp.arima_d, exp.arima_q

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_past_observed = batch[BatchKeys.SYNOP_PAST_Y.value].float()
        if self.config.experiment.use_gfs_data:
            exog_input = batch[BatchKeys.GFS_PAST_X.value].float().numpy()
            exog_future = batch[BatchKeys.GFS_FUTURE_X.value].float().numpy()
            # Remove constant values - Arima can't deduce how they are affecting the model
            same_columns = np.all(exog_input[0][1:] == exog_input[0][:-1], axis=0)
            exog_input = exog_input[:, :, ~same_columns]
            exog_future = exog_future[:, :, ~same_columns]

            arima_model = ARIMA(synop_past_observed[0].numpy(), exog_input[0], (self.p, self.d, self.q))
            model_fit = arima_model.fit()

            return torch.Tensor(model_fit.forecast(steps=self.config.experiment.future_sequence_length,
                                                   exog=exog_future[0])).unsqueeze(0)
        else:
            arima_model = ARIMA(synop_past_observed[0].numpy(), order=(self.p, self.d, self.q))
            model_fit = arima_model.fit()

            return torch.Tensor(model_fit.forecast(steps=self.config.experiment.future_sequence_length)).unsqueeze(0)
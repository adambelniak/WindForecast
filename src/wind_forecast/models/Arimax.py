from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys


class Arimax(pl.LightningModule):
    def __init__(self,  config: Config):
        super(Arimax, self).__init__()
        self.config = config
        exp = self.config.experiment
        self.p, self.d, self.q = exp.arima_p, exp.arima_d, exp.arima_q
        assert exp.batch_size == 0, "Set experiment.batch_size to 0 for ParallelArimax model"
        assert config.experiment.use_gfs_data, "ParallelArimax needs GFS forecasts for modelling"
        self.model_fit = None

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_future_observed = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().numpy()
        gfs_future_features = batch[BatchKeys.GFS_FUTURE_X.value].float().numpy()

        if stage in ['fit']:
            # create array [---series1---xxxxxxx---series2---xxxxxxx---...]
            endog = np.asarray([np.concatenate([series, np.full(synop_future_observed.shape[1], np.nan)])
                                for series in synop_future_observed]).flatten()
            exog = np.asarray([np.concatenate([series, np.full(gfs_future_features.shape[1:], 0)], axis=0)
                                for series in gfs_future_features]).reshape((endog.shape[0], gfs_future_features.shape[-1]))
            self.model_fit = SARIMAX(endog=endog, exog=exog, order=(self.p, self.d, self.q), missing='drop')
            self.model_fit = self.model_fit.fit(maxiter=100)

            return torch.Tensor(np.zeros((synop_future_observed.shape[0], self.config.experiment.future_sequence_length)))
        else:
            assert self.model_fit is not None, "fit stage expected before making predictions"
            synop_past_observed = batch[BatchKeys.SYNOP_PAST_Y.value].float().numpy()
            gfs_past_features = batch[BatchKeys.GFS_PAST_X.value].float().numpy()
            predictions = []

            for index in range(gfs_past_features.shape[0]):
                self.model_fit = self.model_fit.apply(endog=synop_past_observed[index], exog=gfs_past_features[index])

                predictions.append(self.model_fit.forecast(steps=self.config.experiment.future_sequence_length,
                                              exog=gfs_future_features[index]))

            return torch.Tensor(np.asarray(predictions))

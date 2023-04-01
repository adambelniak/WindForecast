from typing import Dict

import torch
from pytorch_lightning import LightningModule
from sklearn.linear_model import Ridge

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
import numpy as np

class LinearRegression(LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert config.experiment.use_gfs_data, "Linear regression needs GFS forecasts for modelling"
        assert config.experiment.batch_size == 0, "Set experiment.batch_size to 0 for LinearRegression model"

        self.regressor = Ridge(max_iter=config.experiment.linear_max_iter,
                               solver='sag',
                               alpha=config.experiment.linear_L2_alpha,
                               tol=1e-6)
        self.model_fit = None

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_future_observed = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().numpy()
        gfs_future_features = batch[BatchKeys.GFS_FUTURE_X.value].float().numpy()
        synop_future_observed = np.reshape(synop_future_observed, (synop_future_observed.shape[0] * synop_future_observed.shape[1]))
        gfs_future_features = np.reshape(gfs_future_features, (gfs_future_features.shape[0] * gfs_future_features.shape[1], gfs_future_features.shape[2]))

        if stage in ['fit']:
            self.model_fit = self.regressor.fit(gfs_future_features, synop_future_observed)

        elif self.model_fit is None:
            assert False, "fit stage expected before making predictions"

        predictions = self.model_fit.predict(gfs_future_features)

        return torch.Tensor(
            np.reshape(predictions, (predictions.shape[0] // self.config.experiment.future_sequence_length,
                                     self.config.experiment.future_sequence_length, 1)))



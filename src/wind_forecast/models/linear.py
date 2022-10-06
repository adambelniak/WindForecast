from typing import Dict

import torch
from pytorch_lightning import LightningModule
from sklearn.linear_model import LinearRegression as LR
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge


class LinearRegression(LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert config.experiment.use_gfs_data, "Linear regression needs GFS forecasts for modelling"

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_past_observed = batch[BatchKeys.SYNOP_PAST_Y.value].float().numpy()
        gfs_past_features = batch[BatchKeys.GFS_PAST_X.value].float().numpy()
        gfs_future_features = batch[BatchKeys.GFS_FUTURE_X.value].float().numpy()
        # batch size 1
        # regressor = LR().fit(gfs_past_features[0], synop_past_observed[0])
        # regressor = ElasticNet().fit(gfs_past_features[0], synop_past_observed[0])
        regressor = Ridge().fit(gfs_past_features[0], synop_past_observed[0])
        return torch.Tensor(regressor.predict(gfs_future_features[0]))


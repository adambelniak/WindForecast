from enum import Enum
from typing import Dict

import torch
from pytorch_lightning import LightningModule
import pandas as pd
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from prophet import Prophet as FBProphet
import logging


class Scope(Enum):
    INSAMPLE = 'insample'
    OUTSAMPLE = 'outsample'


class Prophet(LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        forecasts = []
        for batch_index in range(self.config.experiment.batch_size):
            fit_df = self.prepare_df(batch, batch_index, Scope.INSAMPLE)
            model = FBProphet()
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            for column in [column for column in fit_df.columns if column.startswith("column_")]:
                model.add_regressor(column, standardize=False)
            model.fit(fit_df)

            test_df = self.prepare_df(batch, batch_index, Scope.OUTSAMPLE)
            forecast = model.predict(test_df)
            forecasts.append(forecast['yhat'])

        return torch.Tensor(forecasts)

    def prepare_df(self, batch: Dict[str, torch.Tensor], batch_index: int, scope: Scope):
        if scope == Scope.INSAMPLE:
            exog = batch[BatchKeys.GFS_PAST_X.value][batch_index, :, :].numpy()
            y = batch[BatchKeys.SYNOP_PAST_Y.value][batch_index, :].numpy()
            ds = batch[BatchKeys.DATES_PAST.value][batch_index]
            df = pd.DataFrame(np.concatenate([exog, np.expand_dims(y, -1)], -1), columns=[*[f"column_{column}" for column in range(exog.shape[-1])], 'y'])
            df['ds'] = ds
            return df
        else:
            exog = batch[BatchKeys.GFS_FUTURE_X.value][batch_index, :, :].numpy()
            ds = batch[BatchKeys.DATES_FUTURE.value][batch_index]
            df = pd.DataFrame(exog, columns=[*[f"column_{column}" for column in range(exog.shape[-1])]])
            df['ds'] = ds
            return df
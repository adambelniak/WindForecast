from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from einops import rearrange

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys


class Arimax(pl.LightningModule):
    def __init__(self, config: Config):
        super(Arimax, self).__init__()
        self.config = config
        exp = self.config.experiment
        self.p, self.d, self.q = exp.arima_p, exp.arima_d, exp.arima_q
        assert exp.batch_size == 0, "Set experiment.batch_size to 0 for Arimax model"
        assert config.experiment.use_gfs_data, "Arimax needs GFS forecasts for modelling"
        self.model_fit = None

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_future_observed = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().numpy()
        gfs_future_features = batch[BatchKeys.GFS_FUTURE_X.value].float().numpy()
        synop_past_observed = batch[BatchKeys.SYNOP_PAST_Y.value].float().numpy()
        gfs_past_features = batch[BatchKeys.GFS_PAST_X.value].float().numpy()
        dates_past = batch[BatchKeys.DATES_PAST.value]

        if stage in ['fit']:
            # create long series
            endog = pd.DataFrame({'values': np.asarray([series for series in synop_past_observed]).flatten(),
                                  'dates': np.asarray([series for series in dates_past]).flatten()},
                                 columns=['values', 'dates'])

            exog = pd.DataFrame({'values': np.asarray([index for index in range(rearrange(gfs_past_features, "b s f -> (b s) f").shape[0])]),
                                 'dates': np.asarray([series for series in dates_past]).flatten()},
                                columns=['values', 'dates'])

            endog.drop_duplicates(subset=['dates'], inplace=True)
            exog.drop_duplicates(subset=['dates'], inplace=True)

            endog.sort_values(by='dates', inplace=True)
            exog.sort_values(by='dates', inplace=True)

            first_date, last_date = endog['dates'].values[0], endog['dates'].values[-1]
            date = first_date
            while date < last_date:
                if len(endog[endog['dates'] == date]) == 0:
                    pd.concat([endog, pd.DataFrame([[date, np.nan]], columns=['dates', 'values'])])
                    pd.concat([exog, pd.DataFrame([[date, np.nan]], columns=['dates', 'values'])])
                date += np.timedelta64(1, 'h')

            endog.sort_values(by='dates', inplace=True)
            exog.sort_values(by='dates', inplace=True)

            rearranged_gfs = rearrange(gfs_past_features, "b s f -> (b s) f")
            exog = np.asarray([rearranged_gfs[index] if not np.isnan(index) else np.full((1, *rearranged_gfs.shape[1:]), np.nan) for index in exog['values']])

            self.model_fit = SARIMAX(endog=endog['values'].values, exog=exog,
                                     order=(self.p, self.d, self.q), missing='drop')
            self.model_fit = self.model_fit.fit(maxiter=100)

            return torch.Tensor(
                np.zeros((synop_future_observed.shape[0], self.config.experiment.future_sequence_length)))
        else:
            assert self.model_fit is not None, "fit stage expected before making predictions"

            predictions = []

            for index in range(gfs_past_features.shape[0]):
                self.model_fit = self.model_fit.apply(endog=synop_past_observed[index], exog=gfs_past_features[index])

                predictions.append(self.model_fit.forecast(steps=self.config.experiment.future_sequence_length,
                                                           exog=gfs_future_features[index]))

            return torch.Tensor(np.asarray(predictions))

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.systems.S2SRegressorWithGFSInput import S2SRegressorWithGFSInput


class S2SPredictorWithGFSInput(S2SRegressorWithGFSInput):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)  # type: ignore

        self.predict_results = []

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]
        dates_embeddings = self.get_dates_tensor(dates_inputs, dates_targets)
        batch[BatchKeys.DATES_TENSORS.value] = dates_embeddings
        past_targets = batch[BatchKeys.SYNOP_PAST_Y.value].float().squeeze()

        prediction = self.forward(batch, self.current_epoch, 'test').squeeze()

        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value]
        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]

        if self.categorical_experiment:
            past_targets *= self.classes - 1

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value]
        gfs_past = batch[BatchKeys.GFS_PAST_Y.value]

        all_series = {
            BatchKeys.PREDICTIONS.value: prediction.squeeze() if not self.categorical_experiment else torch.argmax(prediction, dim=-1),
            BatchKeys.SYNOP_PAST_Y.value: past_targets,
            BatchKeys.SYNOP_PAST_X.value: synop_inputs[:, :, self.target_param_index] if self.cfg.experiment.batch_size > 1
            else synop_inputs[:, self.target_param_index],
            BatchKeys.DATES_PAST.value: dates_inputs,
            BatchKeys.DATES_FUTURE.value: dates_targets,
            BatchKeys.GFS_FUTURE_Y.value: gfs_targets.squeeze() if not self.categorical_experiment else gfs_targets.squeeze() * (
                        self.classes - 1),
            BatchKeys.GFS_PAST_Y.value: gfs_past.squeeze() if not self.categorical_experiment else gfs_past.squeeze() * (
                    self.classes - 1)
        }

        results = self.get_results(all_series)

        self.logger.log_metrics(results, step=0)

        self.predict_results = results

    def get_results(self, outputs: Dict) -> Dict:
        past_dates = [pd.to_datetime(d) for d in outputs[BatchKeys.DATES_PAST.value][0]]
        prediction_dates = [pd.to_datetime(d) for d in outputs[BatchKeys.DATES_FUTURE.value][0]]
        all_dates = [*past_dates]
        all_dates.extend(prediction_dates)

        return {
            'truth_series': np.asarray(outputs[BatchKeys.SYNOP_PAST_Y.value].cpu()),
            'prediction': np.asarray(outputs[BatchKeys.PREDICTIONS.value].cpu()),
            'all_dates': all_dates,
            'past_dates': past_dates,
            'prediction_dates': prediction_dates,
            'gfs_targets': np.asarray(outputs[BatchKeys.GFS_FUTURE_Y.value].cpu()),
            'gfs_past': np.asarray(outputs[BatchKeys.GFS_PAST_Y.value].cpu())
        }


from typing import Union, Any, Dict, Optional, List

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from wandb.sdk.wandb_run import Run
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.util.gfs_util import add_param_to_train_params
import pandas as pd

class S2SPredictorWithGFSInput(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()  # type: ignore

        self.logger: Union[LoggerCollection, WandbLogger, Any]
        self.wandb: Run

        self.cfg = cfg

        self.model: LightningModule = instantiate(self.cfg.experiment.model, self.cfg)

        self.predict_results = []
        self.categorical_experiment = self.cfg.experiment.categorical_experiment
        self.classes = self.cfg.experiment.classes
        train_params = self.cfg.experiment.synop_train_features
        target_param = self.cfg.experiment.target_parameter
        all_params = add_param_to_train_params(train_params, target_param)
        feature_names = list(list(zip(*all_params))[1])
        self.target_param_index = [x for x in feature_names].index(target_param)

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

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value]

        all_series = {
            BatchKeys.PREDICTIONS.value: prediction.squeeze() if not self.categorical_experiment else torch.argmax(prediction, dim=-1),
            BatchKeys.SYNOP_PAST_Y.value: past_targets,
            BatchKeys.SYNOP_PAST_X.value: synop_inputs[:, :, self.target_param_index] if self.cfg.experiment.batch_size > 1
            else synop_inputs[:, self.target_param_index],
            BatchKeys.DATES_PAST.value: dates_inputs,
            BatchKeys.DATES_FUTURE.value: dates_targets,
            BatchKeys.GFS_FUTURE_Y.value: gfs_targets.squeeze() if not self.categorical_experiment else gfs_targets.squeeze() * (
                        self.classes - 1)
        }

        results = self.get_results(all_series)

        self.logger.log_metrics(results, step=0)

        self.predict_results = results

    def get_results(self, outputs: Dict) -> Dict:
        all_dates = [pd.to_datetime(pd.Timestamp(d)) for d in outputs[BatchKeys.DATES_PAST.value]]
        prediction_dates = [pd.to_datetime(pd.Timestamp(d)) for d in outputs[BatchKeys.DATES_FUTURE.value]]
        all_dates.extend(prediction_dates)

        return {
            'truth_series': np.asarray(outputs[BatchKeys.SYNOP_PAST_Y.value].cpu()),
            'prediction': np.asarray(outputs[BatchKeys.PREDICTIONS.value].cpu()),
            'all_dates': all_dates,
            'prediction_dates': prediction_dates,
            'gfs_targets': np.asarray(outputs[BatchKeys.GFS_FUTURE_Y.value].cpu())
        }


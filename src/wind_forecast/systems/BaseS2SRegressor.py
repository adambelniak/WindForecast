import copy
import math
from typing import Union, Tuple, List, Any, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics.regression.mean_absolute_error import MeanAbsoluteError
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.regression.mean_squared_error import MeanSquaredError
from pytorch_forecasting.metrics import MASE
from rich import print
from torch.nn import MSELoss
from torch.optim.optimizer import Optimizer
from wandb.sdk.wandb_run import Run

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.util.gfs_util import add_param_to_train_params


class BaseS2SRegressor(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()  # type: ignore

        self.logger: Union[LoggerCollection, WandbLogger, Any]
        self.wandb: Run

        self.cfg = cfg

        self.model: LightningModule = instantiate(self.cfg.experiment.model, self.cfg)

        self.criterion = MSELoss()

        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.train_mase = MASE()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_mase = MASE()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.test_mase = MASE()
        self.test_results = []
        train_params = self.cfg.experiment.synop_train_features
        target_param = self.cfg.experiment.target_parameter
        all_params = add_param_to_train_params(train_params, target_param)
        feature_names = list(list(zip(*all_params))[1])
        self.target_param_index = [x for x in feature_names].index(target_param)

    def get_dates_tensor(self, input_dates, target_dates):
        if self.cfg.experiment.use_time2vec:
            # put day of year and hour as min-max values, they will be embedded via time2vec in model
            input_embed = self.dates_to_min_max_tensor(input_dates)
            target_embed = self.dates_to_min_max_tensor(target_dates)
        else:
            input_embed = self.dates_to_sine_tensor(input_dates)
            target_embed = self.dates_to_sine_tensor(target_dates)

        return input_embed, target_embed

    def dates_to_min_max_tensor(self, dates):
        day_of_year_argument = (183 - np.array(
            [[[pd.to_datetime(d).timetuple().tm_yday] for d in sublist] for sublist in dates])) / 183
        hour_argument = (12 - np.array([[[pd.to_datetime(d).hour] for d in sublist] for sublist in dates])) / 12
        return torch.Tensor(np.concatenate([day_of_year_argument, hour_argument], -1)).to(self.device)

    def dates_to_sine_tensor(self, dates):
        day_of_year_argument = np.array([[[pd.to_datetime(d).timetuple().tm_yday] for d in sublist] for sublist in dates])
        hour_argument = np.array([[[pd.to_datetime(d).hour] for d in sublist] for sublist in dates])
        day_of_year_embed = day_of_year_argument / 365 * 2 * np.pi
        hour_embed = hour_argument / 24 * 2 * np.pi
        return torch.Tensor(np.concatenate([np.sin(day_of_year_embed), np.cos(day_of_year_embed),
                             np.sin(hour_embed), np.cos(hour_embed)], axis=-1)).to(self.device)

    # -----------------------------------------------------------------------------------------------
    # Default PyTorch Lightning hooks
    # -----------------------------------------------------------------------------------------------
    def on_fit_start(self) -> None:
        """
        Hook before `trainer.fit()`.

        Attaches current wandb run to `self.wandb`.
        """
        if isinstance(self.logger, LoggerCollection):
            for logger in self.logger:  # type: ignore
                if isinstance(logger, WandbLogger):
                    self.wandb = logger.experiment  # type: ignore
        elif isinstance(self.logger, WandbLogger):
            self.wandb = self.logger.experiment  # type: ignore

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook on checkpoint saving.

        Adds config and RNG states to the checkpoint file.
        """
        checkpoint['cfg'] = self.cfg

    # ----------------------------------------------------------------------------------------------
    # Optimizers
    # ----------------------------------------------------------------------------------------------
    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:  # type: ignore
        """
        Define system optimization procedure.

        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers.

        Returns
        -------
        Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]
            Single optimizer or a combination of optimizers with learning rate schedulers.
        """
        optimizer: Optimizer = instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters(),
            lr=self.cfg.optim.base_lr,
            _convert_='all'
        )

        if self.cfg.optim.scheduler is not None:
            # if self.cfg.optim.scheduler._target_ == "torch.optim.lr_scheduler.LambdaLR":
            lambda_lr = instantiate(self.cfg.optim.lambda_lr,
                                    warmup_epochs=self.cfg.optim.warmup_epochs,
                                    decay_epochs=self.cfg.optim.decay_epochs,
                                    starting_lr=self.cfg.optim.starting_lr,
                                    base_lr=self.cfg.optim.base_lr,
                                    final_lr=self.cfg.optim.final_lr)

            scheduler: _LRScheduler = instantiate(  # type: ignore
                self.cfg.optim.scheduler,
                optimizer=optimizer,
                lr_lambda=lambda epoch: lambda_lr.transformer_lr_scheduler(epoch),
                _convert_='all',
                verbose=True
            )

            print(optimizer, scheduler)
            return [optimizer], [scheduler]
        else:
            print(optimizer)
            return optimizer

    def _reduce(self, outputs: List[Any], key: str):
        return torch.stack([out[key] for out in outputs]).mean().detach()

    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor], epoch, stage) -> torch.Tensor:
        return self.model(batch, epoch, stage)

    # ----------------------------------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------------------------------
    def calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value of a batch.

        In this simple case just forwards computation to default `self.criterion`.

        Parameters
        ----------
        outputs : torch.Tensor
            Network outputs with shape (batch_size, n_classes).
        targets : torch.Tensor
            Targets (ground-truth labels) with shape (batch_size).

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        if self.cfg.optim.loss == 'rmse':
            return torch.sqrt(self.criterion(outputs, targets))
        elif self.cfg.optim.loss == 'mse':
            return self.criterion(outputs, targets)
        else:
            return self.criterion(outputs, targets)


    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Train on a single batch with loss defined by `self.criterion`.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """

        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]
        dates_embeddings = self.get_dates_tensor(dates_inputs, dates_targets)
        batch[BatchKeys.DATES_TENSORS.value] = dates_embeddings

        outputs = self.forward(batch, self.current_epoch, 'fit').squeeze()
        past_targets = batch[BatchKeys.SYNOP_PAST_Y.value].float().squeeze()
        targets = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().squeeze()
        if self.cfg.experiment.differential_forecast:
            targets = batch[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value].float().squeeze()
            past_targets = batch[BatchKeys.GFS_SYNOP_PAST_DIFF.value].float().squeeze()

        self.train_mse(outputs, targets)
        self.train_mae(outputs, targets)
        if len(targets.shape) == 1:
            self.train_mase(outputs.unsqueeze(0), targets.unsqueeze(0), past_targets.unsqueeze(0))
        else:
            self.train_mase(outputs, targets, past_targets)

        if self.cfg.optim.loss != 'mase':
            loss = self.calculate_loss(outputs, targets)
        else:
            loss = MASE()
            loss = loss(outputs, targets, past_targets)
        return {
            'loss': loss
            # no need to return 'train_mse' here since it is always available as `self.train_mse`
        }

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        Log training metrics.

        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.training_step` with batch metrics.
        """
        step = self.current_epoch + 1

        metrics = {
            'epoch': float(step),
            'train_rmse': math.sqrt(float(self.train_mse.compute().item())),
            'train_mae': float(self.train_mae.compute().item()),
            'train_mase': float(self.train_mase.compute())
        }

        self.train_mse.reset()
        self.train_mae.reset()
        self.train_mase.reset()

        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())

        self.logger.log_metrics(metrics, step=step)

    # ----------------------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------------------
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Compute validation metrics.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]
        dates_embeddings = self.get_dates_tensor(dates_inputs, dates_targets)
        batch[BatchKeys.DATES_TENSORS.value] = dates_embeddings

        outputs = self.forward(batch, self.current_epoch, 'test').squeeze()
        past_targets = batch[BatchKeys.SYNOP_PAST_Y.value].float().squeeze()
        targets = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().squeeze()
        if self.cfg.experiment.differential_forecast:
            targets = batch[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value].float().squeeze()
            past_targets = batch[BatchKeys.GFS_SYNOP_PAST_DIFF.value].float().squeeze()

        self.val_mse(outputs, targets)
        self.val_mae(outputs, targets)
        if len(targets.shape) == 1:
            self.val_mase(outputs.unsqueeze(0), targets.unsqueeze(0), past_targets.unsqueeze(0))
        else:
            self.val_mase(outputs, targets, past_targets)

        return {
            # 'additional_metric': ...
            # no need to return 'val_mse' here since it is always available as `self.val_mse`
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        """
        Log validation metrics.

        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.validation_step` with batch metrics.
        """
        step = self.current_epoch + 1 if not self.trainer.sanity_checking else self.current_epoch  # type: ignore

        metrics = {
            'epoch': float(step),
            'val_rmse': math.sqrt(float(self.val_mse.compute().item())),
            'val_mae': float(self.val_mae.compute().item()),
            'val_mase': float(self.val_mase.compute())
        }

        self.val_mse.reset()
        self.val_mae.reset()
        self.val_mase.reset()

        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())

        self.logger.log_metrics(metrics, step=step)
        self.log("ptl/val_rmse", metrics['val_rmse'])
        self.log("ptl/val_mase", metrics['val_mase'])

    # ----------------------------------------------------------------------------------------------
    # Test
    # ----------------------------------------------------------------------------------------------
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Compute test metrics.

        Parameters
        ----------
        batch : Batch
            Test batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]
        dates_embeddings = self.get_dates_tensor(dates_inputs, dates_targets)
        batch[BatchKeys.DATES_TENSORS.value] = dates_embeddings

        outputs = self.forward(batch, self.current_epoch, 'test').squeeze()
        past_targets = batch[BatchKeys.SYNOP_PAST_Y.value].float().squeeze()
        targets = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().squeeze()

        if self.cfg.experiment.differential_forecast:
            targets = batch[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value].float().squeeze()
            past_targets = batch[BatchKeys.GFS_SYNOP_PAST_DIFF.value].float().squeeze()

        self.test_mse(outputs, targets)
        self.test_mae(outputs, targets)
        if len(targets.shape) == 1:
            self.test_mase(outputs.unsqueeze(0), targets.unsqueeze(0), past_targets.unsqueeze(0))
        else:
            self.test_mase(outputs, targets, past_targets)

        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value]
        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]

        return {BatchKeys.SYNOP_FUTURE_Y.value: targets,
                BatchKeys.PREDICTIONS.value: outputs.squeeze(),
                BatchKeys.SYNOP_PAST_Y.value: past_targets[:],
                BatchKeys.SYNOP_PAST_X.value: synop_inputs[:, :, self.target_param_index] if self.cfg.experiment.batch_size > 1
                else synop_inputs[:, self.target_param_index],
                BatchKeys.DATES_PAST.value: dates_inputs,
                BatchKeys.DATES_FUTURE.value: dates_targets
                }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """
        Log test metrics.

        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.test_step` with batch metrics.
        """
        step = self.current_epoch + 1 if not self.trainer.sanity_checking else self.current_epoch  # type: ignore

        metrics_and_plot_results = self.get_metrics_and_plot_results(step, outputs)

        self.test_mse.reset()
        self.test_mae.reset()
        self.test_mase.reset()

        self.logger.log_metrics(metrics_and_plot_results, step=step)

        self.test_results = metrics_and_plot_results

    def get_metrics_and_plot_results(self, step: int, outputs: List[Any]) -> Dict:
        series = self.get_series_from_outputs(outputs)

        predictions = series[BatchKeys.PREDICTIONS.value]
        rmse_by_step = np.sqrt(np.mean(np.power(np.subtract(series[BatchKeys.PREDICTIONS.value], series[BatchKeys.SYNOP_FUTURE_Y.value]), 2), axis=0))
        mase_by_step = self.get_mase_by_step(series[BatchKeys.SYNOP_PAST_Y.value], predictions)

        # for plots
        plot_truth_series = []
        plot_prediction_series = []
        plot_all_dates = []
        plot_prediction_dates = []
        for index in np.random.choice(np.arange(len(predictions)),
                                      min(40, len(predictions)), replace=False):
            sample_all_dates = [pd.to_datetime(pd.Timestamp(d)) for d in series[BatchKeys.DATES_PAST.value][index]]
            sample_prediction_dates = [pd.to_datetime(pd.Timestamp(d)) for d in series[BatchKeys.DATES_FUTURE.value][index]]
            sample_all_dates.extend(sample_prediction_dates)

            plot_all_dates.append(sample_all_dates)
            plot_prediction_dates.append(sample_prediction_dates)

            plot_prediction_series.append(predictions[index])
            plot_truth_series.append(np.concatenate([series[BatchKeys.SYNOP_PAST_Y.value][index], series[BatchKeys.SYNOP_FUTURE_Y.value][index]], 0).tolist())

        return {
            'epoch': float(step),
            'test_rmse': math.sqrt(float(self.test_mse.compute().item())),
            'test_mae': float(self.test_mae.compute().item()),
            'test_mase': float(self.test_mase.compute()),
            'rmse_by_step': rmse_by_step,
            'mase_by_step': mase_by_step,
            'plot_truth': plot_truth_series,
            'plot_prediction': plot_prediction_series,
            'plot_all_dates': plot_all_dates,
            'plot_prediction_dates': plot_prediction_dates
        }

    def get_series_from_outputs(self, outputs: List[Any]) -> Dict:
        if self.cfg.experiment.batch_size > 1:
            prediction_series = [item.cpu() for sublist in [x[BatchKeys.PREDICTIONS.value] for x in outputs] for item in sublist]
            synop_future_y_series = [item.cpu() for sublist in [x[BatchKeys.SYNOP_FUTURE_Y.value] for x in outputs] for item in sublist]
            synop_past_y_series = [item.cpu() for sublist in [x[BatchKeys.SYNOP_PAST_Y.value] for x in outputs] for item in sublist]
            past_dates = [item for sublist in [x[BatchKeys.DATES_PAST.value] for x in outputs] for item in sublist]
            future_dates = [item for sublist in [x[BatchKeys.DATES_FUTURE.value] for x in outputs] for item in sublist]
        else:
            prediction_series = [item.cpu() for item in [x[BatchKeys.PREDICTIONS.value] for x in outputs]]
            synop_future_y_series = [item.cpu() for item in [x[BatchKeys.SYNOP_FUTURE_Y.value] for x in outputs]]
            synop_past_y_series = [item.cpu() for item in [x[BatchKeys.SYNOP_PAST_Y.value] for x in outputs]]
            # mistery - why there are 1-element tuples?
            past_dates = [item[0] for item in [x[BatchKeys.DATES_PAST.value] for x in outputs]]
            future_dates = [item[0] for item in [x[BatchKeys.DATES_FUTURE.value] for x in outputs]]
        prediction_series = np.asarray([np.asarray(el) for el in prediction_series])
        synop_future_y_series = np.asarray([np.asarray(el) for el in synop_future_y_series])
        synop_past_y_series = np.asarray([np.asarray(el) for el in synop_past_y_series])

        return {
            BatchKeys.PREDICTIONS.value: prediction_series,
            BatchKeys.SYNOP_FUTURE_Y.value: synop_future_y_series,
            BatchKeys.SYNOP_PAST_Y.value: synop_past_y_series,
            BatchKeys.DATES_PAST.value: past_dates,
            BatchKeys.DATES_FUTURE.value: future_dates
        }

    def get_mase_by_step(self, truth_series, prediction_series):
        mase_by_step = []
        for step in range(prediction_series.shape[-1]):
            mase_by_step.append(
                (abs(prediction_series[:, :step + 1] - truth_series[:, :step + 1]).mean() /
                 abs(truth_series[:, :-1] - truth_series[:, 1:]).mean()).mean())
        return mase_by_step
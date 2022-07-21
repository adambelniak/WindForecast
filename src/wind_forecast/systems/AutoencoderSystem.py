from __future__ import annotations

import math
from typing import Any, Tuple, Union, List

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.regression.mean_squared_error import MeanSquaredError
from rich import print
from torch.nn import MSELoss
from torch.optim.optimizer import Optimizer
from wandb.sdk.wandb_run import Run

from wind_forecast.config.register import Config


class AutoencoderSystem(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()  # type: ignore

        self.logger: Union[LoggerCollection, WandbLogger, Any]
        self.wandb: Run

        self.cfg = cfg

        self.model: LightningModule = instantiate(
            self.cfg.experiment.model,
            self.cfg)

        self.criterion = MSELoss()

        # Metrics
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.test_results = []

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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
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
            _convert_='all'
        )

        if self.cfg.optim.scheduler is not None:
            # if self.cfg.optim.scheduler._target_ == "torch.optim.lr_scheduler.LambdaLR":
            lambda_lr = instantiate(self.cfg.optim.lambda_lr,
                                    warmup_epochs=self.cfg.optim.warmup_epochs,
                                    decay_epochs=self.cfg.optim.decay_epochs,
                                    starting_lr=self.cfg.optim.starting_lr,
                                    base_lr=self.cfg.optim.lr,
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

    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
        return self.criterion(outputs, targets.float())

    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:  # type: ignore
        """
        Train on a single batch with loss defined by `self.criterion`.

        Parameters
        ----------
        batch : list[torch.Tensor]
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        inputs = batch.float()
        outputs = self.forward(inputs)
        loss = self.calculate_loss(outputs, inputs)
        self.train_mse(outputs, inputs)

        return {
            'loss': loss,
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
        }

        self.train_mse.reset()

        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())

        self.logger.log_metrics(metrics, step=step)

    def _reduce(self, outputs: List[Any], key: str):
        return torch.stack([out[key] for out in outputs]).mean().detach()

    # ----------------------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------------------
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:  # type: ignore
        """
        Compute validation metrics.

        Parameters
        ----------
        batch : list[torch.Tensor]
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        inputs = batch.float()
        outputs = self.forward(inputs)

        self.val_mse(outputs, inputs)

        return {
            # 'additional_metric': ...
            # no need to return 'val_mse' here since it is always available as `self.val_mse`
        }

    def validation_epoch_end(self, outputs: List[Any]) -> dict[str, Any]:
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
        }

        self.val_mse.reset()

        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())

        self.logger.log_metrics(metrics, step=step)
        self.log("ptl/val_loss", metrics['val_rmse'])

    # ----------------------------------------------------------------------------------------------
    # Test
    # ----------------------------------------------------------------------------------------------
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:  # type: ignore
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
        inputs = batch.float()
        outputs = self.forward(inputs)

        self.test_mse(outputs, inputs)

        return {}

    def test_epoch_end(self, outputs: List[Any]) -> dict[str, Any]:
        """
        Log test metrics.

        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.test_step` with batch metrics.
        """
        step = self.current_epoch + 1 if not self.trainer.sanity_checking else self.current_epoch  # type: ignore

        metrics = {
            'epoch': float(step),
            'test_rmse': math.sqrt(float(self.test_mse.compute().item())),
        }

        self.test_mse.reset()

        # Average additional metrics over all batches
        # for key in outputs[0]:
        #     metrics[key] = float(self._reduce(outputs, key).item())

        self.logger.log_metrics(metrics, step=step)

        self.test_results = {}


from __future__ import annotations

from typing import List, Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.systems.BaseS2SRegressor import BaseS2SRegressor


class S2SRegressorWithTFWithCMAXWithGFS(BaseS2SRegressor):

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, gfs_targets, cmax_inputs: torch.Tensor, targets: torch.Tensor, epoch, stage, gfs_inputs: torch.Tensor = None, cmax_targets: torch.Tensor = None) -> torch.Tensor:
        if cmax_targets is None:
            if gfs_inputs is None:
                return self.model(x.float(), None, gfs_targets.float(), cmax_inputs.float(), targets.float(), epoch, stage)
            return self.model(x.float(), gfs_inputs.float(), gfs_targets.float(), cmax_inputs.float(), targets.float(), epoch, stage)
        else:
            if gfs_inputs is None:
                return self.model(x.float(), None, gfs_targets.float(), cmax_inputs.float(), targets.float(), cmax_targets.float(), epoch, stage)
            return self.model(x.float(), gfs_inputs.float(), gfs_targets.float(), cmax_inputs.float(), targets.float(), cmax_targets.float(), epoch, stage)

    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Train on a single batch with loss defined by `self.criterion`.

        Parameters
        ----------
        batch : List[torch.Tensor]
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Metric values for a given batch.
        """

        if self.cfg.experiment.use_all_gfs_as_input:
            inputs, gfs_inputs, gfs_targets, synop_targets, targets, targets_dates, inputs_dates = batch[0]
            if self.cfg.experiment.use_future_cmax:
                cmax_inputs, cmax_targets = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'fit', gfs_inputs, cmax_targets)
            else:
                cmax_inputs = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'fit', gfs_inputs)

        else:
            inputs, gfs_targets, synop_targets, targets, targets_dates, inputs_dates = batch[0]
            if self.cfg.experiment.use_future_cmax:
                cmax_inputs, cmax_targets = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'fit', cmax_targets=cmax_targets)
            else:
                cmax_inputs = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'fit')

        loss = self.calculate_loss(outputs, targets.float())
        self.train_mse(outputs, targets)
        self.train_mae(outputs, targets)

        return {
            'loss': loss,
            # no need to return 'train_mse' here since it is always available as `self.train_mse`
        }

    # ----------------------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------------------
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Compute validation metrics.

        Parameters
        ----------
        batch : List[torch.Tensor]
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        Dict[str, torch.Tensor]
            Metric values for a given batch.
        """

        if self.cfg.experiment.use_all_gfs_as_input:
            inputs, gfs_inputs, gfs_targets, synop_targets, targets, targets_dates, inputs_dates = batch[0]
            if self.cfg.experiment.use_future_cmax:
                cmax_inputs, cmax_targets = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'test',
                                       gfs_inputs, cmax_targets)
            else:
                cmax_inputs = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'test',
                                       gfs_inputs)

        else:
            inputs, gfs_targets, synop_targets, targets, targets_dates, inputs_dates = batch[0]
            if self.cfg.experiment.use_future_cmax:
                cmax_inputs, cmax_targets = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'test',
                                       cmax_targets=cmax_targets)
            else:
                cmax_inputs = batch[1]
                outputs = self.forward(inputs, gfs_targets, cmax_inputs, synop_targets, self.current_epoch, 'test')

        self.val_mse(outputs, targets.float())
        self.val_mae(outputs, targets.float())

        return {
            # 'additional_metric': ...
            # no need to return 'val_mse' here since it is always available as `self.val_mse`
        }

    # ----------------------------------------------------------------------------------------------
    # Test
    # ----------------------------------------------------------------------------------------------
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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
        Dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        if self.cfg.experiment.use_all_gfs_as_input:
            synop_inputs, all_gfs_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch[0]
            if self.cfg.experiment.use_future_cmax:
                cmax_inputs, cmax_targets = batch[1]
                outputs = self.forward(synop_inputs, gfs_targets, cmax_inputs, all_synop_targets, self.current_epoch, 'test',
                                       all_gfs_inputs, cmax_targets)
            else:
                cmax_inputs = batch[1]
                outputs = self.forward(synop_inputs, gfs_targets, cmax_inputs, all_synop_targets, self.current_epoch, 'test',
                                       all_gfs_inputs)

        else:
            synop_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch[0]
            if self.cfg.experiment.use_future_cmax:
                cmax_inputs, cmax_targets = batch[1]
                outputs = self.forward(synop_inputs, gfs_targets, cmax_inputs, all_synop_targets, self.current_epoch, 'test',
                                       cmax_targets=cmax_targets)
            else:
                cmax_inputs = batch[1]
                outputs = self.forward(synop_inputs, gfs_targets, cmax_inputs, all_synop_targets, self.current_epoch, 'test')

        self.test_mse(outputs, targets.float())
        self.test_mae(outputs, targets.float())

        return {'labels': targets,
                'output': outputs,
                'input': synop_inputs[:, :, self.target_param_index],
                'targets_dates': targets_dates,
                'inputs_dates': inputs_dates
                }

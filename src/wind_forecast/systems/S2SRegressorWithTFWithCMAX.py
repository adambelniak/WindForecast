from __future__ import annotations

from typing import Dict, List

import torch

from wind_forecast.systems.BaseS2SRegressor import BaseS2SRegressor


class S2SRegressorWithTFWithCMAX(BaseS2SRegressor):
    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, cmax_inputs: torch.Tensor, targets: torch.Tensor, epoch, stage,
                cmax_targets: torch.Tensor = None) -> torch.Tensor:
        if cmax_targets is None:
            return self.model(x.float(), cmax_inputs.float(), targets.float(), epoch, stage)
        else:
            return self.model(x.float(), cmax_inputs.float(), targets.float(), cmax_targets.float(), epoch, stage)

    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
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

        inputs, synop_targets, targets, targets_dates, inputs_dates = batch[0]
        if self.cfg.experiment.use_future_cmax:
            cmax_inputs, cmax_targets = batch[1]
            outputs = self.forward(inputs, cmax_inputs, synop_targets, self.current_epoch, 'fit', cmax_targets)

        else:
            cmax_inputs = batch[1]
            outputs = self.forward(inputs, cmax_inputs, synop_targets, self.current_epoch, 'fit')

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
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
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

        inputs, synop_targets, targets, targets_dates, inputs_dates = batch[0]

        if self.cfg.experiment.use_future_cmax:
            cmax_inputs, cmax_targets = batch[1]
            outputs = self.forward(inputs, cmax_inputs, synop_targets, self.current_epoch, 'test', cmax_targets)

        else:
            cmax_inputs = batch[1]
            outputs = self.forward(inputs, cmax_inputs, synop_targets, self.current_epoch, 'test')

        self.val_mse(outputs, targets.float())
        self.val_mae(outputs, targets.float())

        return {
            # 'additional_metric': ...
            # no need to return 'val_mse' here since it is always available as `self.val_mse`
        }

    # ----------------------------------------------------------------------------------------------
    # Test
    # ----------------------------------------------------------------------------------------------
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:  # type: ignore
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
        inputs, synop_targets, targets, targets_dates, inputs_dates = batch[0]

        if self.cfg.experiment.use_future_cmax:
            cmax_inputs, cmax_targets = batch[1]
            outputs = self.forward(inputs, cmax_inputs, synop_targets, self.current_epoch, 'test', cmax_targets)

        else:
            cmax_inputs = batch[1]
            outputs = self.forward(inputs, cmax_inputs, synop_targets, self.current_epoch, 'test')

        self.test_mse(outputs, targets.float())
        self.test_mae(outputs, targets.float())

        return {'labels': targets,
                'output': outputs,
                'input': inputs[:, :, self.target_param_index],
                'targets_dates': targets_dates,
                'inputs_dates': inputs_dates
                }


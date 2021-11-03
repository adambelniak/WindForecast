from __future__ import annotations

from typing import List, Dict

import torch

from wind_forecast.systems.BaseS2SRegressor import BaseS2SRegressor


class S2SRegressorWithTFWithGFSInput(BaseS2SRegressor):
    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, gfs_targets: torch.Tensor, targets: torch.Tensor, epoch, stage, gfs_inputs: torch.Tensor = None,
                dates_embeddings: torch.Tensor = None) -> torch.Tensor:
        if gfs_inputs is None:
            return self.model(x.float(), None, gfs_targets.float(), targets.float(), epoch, stage, dates_embeddings)
        return self.model(x.float(), gfs_inputs.float(), gfs_targets.float(), targets.float(), epoch, stage, dates_embeddings)

    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
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
        if self.cfg.experiment.use_all_gfs_as_input:
            synop_inputs, all_gfs_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'fit',
                                       all_gfs_inputs, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'fit', all_gfs_inputs)
        else:
            synop_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'fit',
                                       None, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'fit')

        loss = self.calculate_loss(outputs, targets.float())
        self.train_mse(outputs, targets)
        self.train_mae(outputs, targets)

        return {
            'loss': loss
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
        batch : list[torch.Tensor]
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        if self.cfg.experiment.use_all_gfs_as_input:
            synop_inputs, all_gfs_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test',
                                       all_gfs_inputs, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test',
                                       all_gfs_inputs)
        else:
            synop_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test',
                                       None, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test')

        self.val_mse(outputs.squeeze(), targets.float().squeeze())
        self.val_mae(outputs.squeeze(), targets.float().squeeze())

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
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        if self.cfg.experiment.use_all_gfs_as_input:
            synop_inputs, all_gfs_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test',
                                       all_gfs_inputs, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test',
                                       all_gfs_inputs)
        else:
            synop_inputs, gfs_targets, all_synop_targets, targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test',
                                       None, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, self.current_epoch, 'test')

        self.test_mse(outputs.squeeze(), targets.float().squeeze())
        self.test_mae(outputs.squeeze(), targets.float().squeeze())

        return {'labels': targets,
                'output': outputs,
                'input': synop_inputs[:, :, self.target_param_index],
                'targets_dates': targets_dates,
                'inputs_dates': inputs_dates
                }

from __future__ import annotations

import copy
import math
from typing import List, Dict, Any

import torch

from wind_forecast.systems.BaseS2SRegressor import BaseS2SRegressor


class S2SRegressorWithTFWithGFSInput(BaseS2SRegressor):
    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, gfs_targets: torch.Tensor, targets: torch.Tensor, all_gfs_targets: torch.Tensor, epoch, stage, gfs_inputs: torch.Tensor = None,
                dates_embeddings: torch.Tensor = None) -> torch.Tensor:
        if gfs_inputs is None:
            return self.model(x.float(), None, gfs_targets.float(), targets.float(), all_gfs_targets.float(), epoch, stage, dates_embeddings)
        return self.model(x.float(), gfs_inputs.float(), gfs_targets.float(), targets.float(), all_gfs_targets.float(), epoch, stage, dates_embeddings)

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
            synop_inputs, all_gfs_inputs, gfs_targets, all_synop_targets, targets, all_gfs_targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'fit',
                                       all_gfs_inputs, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'fit', all_gfs_inputs)
        else:
            synop_inputs, gfs_targets, all_synop_targets, targets, all_gfs_targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'fit',
                                       None, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'fit')

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
            synop_inputs, all_gfs_inputs, gfs_targets, all_synop_targets, targets, all_gfs_targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test',
                                       all_gfs_inputs, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test',
                                       all_gfs_inputs)
        else:
            synop_inputs, gfs_targets, all_synop_targets, targets, all_gfs_targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test',
                                       None, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test')

        self.val_mse(outputs.squeeze(), targets.float().squeeze())
        self.val_mae(outputs.squeeze(), targets.float().squeeze())

        return {
            # 'additional_metric': ...
            # no need to return 'val_mse' here since it is always available as `self.val_mse`
        }

    def test_epoch_end(self, outputs: List[Any]) -> None:
        """
        Log test metrics.

        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.test_step` with batch metrics.
        """
        step = self.current_epoch + 1 if not self.trainer.running_sanity_check else self.current_epoch  # type: ignore

        metrics = {
            'epoch': float(step),
            'test_rmse': math.sqrt(float(self.test_mse.compute().item())),
            'test_mae': float(self.test_mae.compute().item())
        }

        self.test_mse.reset()
        self.test_mae.reset()

        self.logger.log_metrics(metrics, step=step)

        # save results to view
        labels = [item for sublist in [x['labels'] for x in outputs] for item in sublist]

        inputs_dates = [item for sublist in [x['inputs_dates'] for x in outputs] for item in sublist]

        labels_dates = [item for sublist in [x['targets_dates'] for x in outputs] for item in sublist]

        out = [item for sublist in [x['output'] for x in outputs] for item in sublist]

        inputs = [item for sublist in [x['input'] for x in outputs] for item in sublist]

        gfs_targets = [item for sublist in [x['gfs_targets'] for x in outputs] for item in sublist]

        self.test_results = {'labels': copy.deepcopy(labels),
                             'output': copy.deepcopy(out),
                             'inputs': copy.deepcopy(inputs),
                             'inputs_dates': copy.deepcopy(inputs_dates),
                             'targets_dates': copy.deepcopy(labels_dates),
                             'gfs_targets': copy.deepcopy(gfs_targets)}

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
            synop_inputs, all_gfs_inputs, gfs_targets, all_synop_targets, targets, all_gfs_targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test',
                                       all_gfs_inputs, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test',
                                       all_gfs_inputs)
        else:
            synop_inputs, gfs_targets, all_synop_targets, targets, all_gfs_targets, targets_dates, inputs_dates = batch
            if self.cfg.experiment.with_dates_inputs:
                dates_embeddings = self.get_dates_embeddings(inputs_dates, targets_dates)

                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test',
                                       None, dates_embeddings)
            else:
                outputs = self.forward(synop_inputs, gfs_targets, all_synop_targets, all_gfs_targets, self.current_epoch, 'test')

        self.test_mse(outputs.squeeze(), targets.float().squeeze())
        self.test_mae(outputs.squeeze(), targets.float().squeeze())

        return {'labels': targets,
                'output': outputs,
                'input': synop_inputs[:, :, self.target_param_index],
                'targets_dates': targets_dates,
                'inputs_dates': inputs_dates,
                'gfs_targets': gfs_targets
                }

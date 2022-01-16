from __future__ import annotations

import copy
import math
from typing import List, Dict, Any

import torch

from wind_forecast.consts import BatchKeys
from wind_forecast.systems.BaseS2SRegressor import BaseS2SRegressor


class S2SRegressorWithGFSInput(BaseS2SRegressor):
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
        if self.cfg.experiment.with_dates_inputs:
            dates_inputs = batch[BatchKeys.DATES_INPUTS.value]
            dates_targets = batch[BatchKeys.DATES_TARGETS.value]
            dates_embeddings = self.get_dates_tensor(dates_inputs, dates_targets)
            batch[BatchKeys.DATES_TENSORS.value] = dates_embeddings

        outputs = self.forward(batch, self.current_epoch, 'test')
        targets = batch[BatchKeys.SYNOP_TARGETS.value]

        self.test_mse(outputs.squeeze(), targets.float().squeeze())
        self.test_mae(outputs.squeeze(), targets.float().squeeze())

        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value]
        if self.cfg.experiment.with_dates_inputs:
            dates_inputs = batch[BatchKeys.DATES_INPUTS.value]
            dates_targets = batch[BatchKeys.DATES_TARGETS.value]
        else:
            dates_inputs = None
            dates_targets = None

        gfs_targets = batch[BatchKeys.GFS_TARGETS.value]

        return {BatchKeys.SYNOP_TARGETS.value: targets,
                'output': outputs,
                BatchKeys.SYNOP_INPUTS.value: synop_inputs[:, :, self.target_param_index],
                BatchKeys.DATES_INPUTS.value: dates_inputs,
                BatchKeys.DATES_TARGETS.value: dates_targets,
                BatchKeys.GFS_TARGETS.value: gfs_targets
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

        metrics = {
            'epoch': float(step),
            'test_rmse': math.sqrt(float(self.test_mse.compute().item())),
            'test_mae': float(self.test_mae.compute().item())
        }

        self.test_mse.reset()
        self.test_mae.reset()

        self.logger.log_metrics(metrics, step=step)

        # save results to view
        labels = [item for sublist in [x[BatchKeys.SYNOP_TARGETS.value] for x in outputs] for item in sublist]

        out = [item for sublist in [x['output'] for x in outputs] for item in sublist]

        inputs = [item for sublist in [x[BatchKeys.SYNOP_INPUTS.value] for x in outputs] for item in sublist]

        gfs_targets = [item for sublist in [x[BatchKeys.GFS_TARGETS.value] for x in outputs] for item in sublist]

        if self.cfg.experiment.with_dates_inputs:
            inputs_dates = [item for sublist in [x[BatchKeys.DATES_INPUTS.value] for x in outputs] for item in sublist]
            labels_dates = [item for sublist in [x[BatchKeys.DATES_TARGETS.value] for x in outputs] for item in sublist]
        else:
            inputs_dates = None
            labels_dates = None

        self.test_results = {'labels': copy.deepcopy(labels),
                             'output': copy.deepcopy(out),
                             'inputs': copy.deepcopy(inputs),
                             'inputs_dates': copy.deepcopy(inputs_dates),
                             'targets_dates': copy.deepcopy(labels_dates),
                             'gfs_targets': copy.deepcopy(gfs_targets)}


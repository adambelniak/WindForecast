from __future__ import annotations

import copy
import math
from typing import List, Dict, Any

import torch

from wind_forecast.consts import BatchKeys
from wind_forecast.systems.BaseS2SRegressor import BaseS2SRegressor

"""
Created for raw GFS performance test
"""
class S2SGFSRegressor(BaseS2SRegressor):
    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return {}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

    # ----------------------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------------------
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return {}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        pass

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
        outputs = batch[BatchKeys.GFS_FUTURE_Y.value]
        targets = batch[BatchKeys.SYNOP_FUTURE_Y.value]
        synop_past_targets = batch[BatchKeys.SYNOP_PAST_Y.value]

        self.test_mse(outputs.squeeze(), targets.float().squeeze())
        self.test_mae(outputs.squeeze(), targets.float().squeeze())
        self.test_mase(outputs.squeeze(), targets.float().squeeze(), synop_past_targets)

        return {BatchKeys.SYNOP_FUTURE_Y.value: targets,
                'output': outputs,
                BatchKeys.SYNOP_PAST_Y.value: synop_past_targets[:, :],
                BatchKeys.DATES_PAST.value: dates_inputs,
                BatchKeys.DATES_FUTURE.value: dates_targets,
                BatchKeys.GFS_FUTURE_Y.value: outputs}

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
            'test_mae': float(self.test_mae.compute().item()),
            'test_mase': float(self.test_mase.compute())
        }

        self.test_mse.reset()
        self.test_mae.reset()
        self.test_mase.reset()

        # save results to view
        labels = [item for sublist in [x[BatchKeys.SYNOP_FUTURE_Y.value] for x in outputs] for item in sublist]

        out = [item for sublist in [x['output'] for x in outputs] for item in sublist]

        inputs = [item for sublist in [x[BatchKeys.SYNOP_PAST_Y.value] for x in outputs] for item in sublist]

        gfs_targets = [item for sublist in [x[BatchKeys.GFS_FUTURE_Y.value] for x in outputs] for item in sublist]

        inputs_dates = [item for sublist in [x[BatchKeys.DATES_PAST.value] for x in outputs] for item in sublist]
        labels_dates = [item for sublist in [x[BatchKeys.DATES_FUTURE.value] for x in outputs] for item in sublist]

        self.test_results = {'labels': copy.deepcopy(labels),
                             'output': copy.deepcopy(out),
                             'inputs': copy.deepcopy(inputs),
                             'inputs_dates': copy.deepcopy(inputs_dates),
                             'targets_dates': copy.deepcopy(labels_dates),
                             'gfs_targets': copy.deepcopy(gfs_targets)}

        self.logger.log_metrics(metrics, step=step)

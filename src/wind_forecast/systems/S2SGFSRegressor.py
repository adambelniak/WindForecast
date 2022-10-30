from __future__ import annotations

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
                BatchKeys.PREDICTIONS.value: outputs.squeeze(),
                BatchKeys.SYNOP_PAST_Y.value: synop_past_targets[:, :],
                BatchKeys.DATES_PAST.value: dates_inputs,
                BatchKeys.DATES_FUTURE.value: dates_targets,
                BatchKeys.GFS_FUTURE_Y.value: outputs.squeeze()}

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

        self.test_results = metrics_and_plot_results

        self.logger.log_metrics(metrics_and_plot_results, step=step)

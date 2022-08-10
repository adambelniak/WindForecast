from __future__ import annotations

import copy
import math
from typing import List, Dict, Any
import numpy as np
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
        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]
        dates_embeddings = self.get_dates_tensor(dates_inputs, dates_targets)
        batch[BatchKeys.DATES_TENSORS.value] = dates_embeddings

        outputs = self.forward(batch, self.current_epoch, 'test')
        past_targets = batch[BatchKeys.SYNOP_PAST_Y.value].float().squeeze()
        targets = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().squeeze()

        if self.cfg.experiment.differential_forecast:
            targets = batch[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value].float().squeeze()
            past_targets = batch[BatchKeys.GFS_SYNOP_PAST_DIFF.value].float().squeeze()

        self.test_mse(outputs.squeeze(), targets)
        self.test_mae(outputs.squeeze(), targets)
        if self.cfg.experiment.batch_size == 1:
            self.test_mase(outputs, targets.unsqueeze(0), past_targets.unsqueeze(0))
        else:
            self.test_mase(outputs, targets, past_targets)

        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value]

        return {BatchKeys.SYNOP_FUTURE_Y.value: batch[BatchKeys.SYNOP_FUTURE_Y.value].float().squeeze(),
                'output': outputs.squeeze(),
                BatchKeys.SYNOP_PAST_Y.value: batch[BatchKeys.SYNOP_PAST_Y.value].float().squeeze()[:],
                BatchKeys.DATES_PAST.value: dates_inputs,
                BatchKeys.DATES_FUTURE.value: dates_targets,
                BatchKeys.GFS_FUTURE_Y.value: gfs_targets.squeeze()
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
        base_metrics = super().get_metrics_and_plot_results(step, outputs)
        if self.cfg.experiment.batch_size > 1:
            output_series = [item.cpu() for sublist in [x['output'] for x in outputs] for item in sublist]
            gfs_targets = [item.cpu() for sublist in [x[BatchKeys.GFS_FUTURE_Y.value] for x in outputs] for item in sublist]
        else:
            output_series = [item.cpu() for item in [x['output'] for x in outputs]]
            gfs_targets = [item.cpu() for item in [x[BatchKeys.GFS_FUTURE_Y.value] for x in outputs]]

        plot_gfs_targets = []
        for index in np.random.choice(np.arange(len(output_series)),
                                      min(40, len(output_series)), replace=False):
            plot_gfs_targets.append(gfs_targets[index])

        base_metrics['plot_gfs_targets'] = plot_gfs_targets
        return base_metrics

from __future__ import annotations
import pandas as pd
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

        outputs = self.forward(batch, self.current_epoch, 'test').squeeze()
        past_targets = batch[BatchKeys.SYNOP_PAST_Y.value].float().squeeze()
        targets = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().squeeze()

        if self.cfg.experiment.differential_forecast:
            targets = batch[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value].float().squeeze()
            past_targets = batch[BatchKeys.GFS_SYNOP_PAST_DIFF.value].float().squeeze()

        self.test_mse(outputs.squeeze(), targets)
        self.test_mae(outputs.squeeze(), targets)
        if len(targets.shape) == 1:
            self.test_mase(outputs.unsqueeze(0), targets.unsqueeze(0), past_targets.unsqueeze(0))
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
        if self.cfg.experiment.batch_size > 1:
            output_series = [item.cpu() for sublist in [x['output'] for x in outputs] for item in sublist]
            labels_series = [item.cpu() for sublist in [x[BatchKeys.SYNOP_FUTURE_Y.value] for x in outputs] for item in sublist]
            gfs_targets = [item.cpu() for sublist in [x[BatchKeys.GFS_FUTURE_Y.value] for x in outputs] for item in sublist]
            past_truth_series = [item.cpu() for sublist in [x[BatchKeys.SYNOP_PAST_Y.value] for x in outputs] for item in sublist]
            inputs_dates = [item for sublist in [x[BatchKeys.DATES_PAST.value] for x in outputs] for item in sublist]
            labels_dates = [item for sublist in [x[BatchKeys.DATES_FUTURE.value] for x in outputs] for item in sublist]
        else:
            output_series = [item.cpu() for item in [x['output'] for x in outputs]]
            labels_series = [item.cpu() for item in [x[BatchKeys.SYNOP_FUTURE_Y.value] for x in outputs]]
            gfs_targets = [item.cpu() for item in [x[BatchKeys.GFS_FUTURE_Y.value] for x in outputs]]
            past_truth_series = [item.cpu() for item in [x[BatchKeys.SYNOP_PAST_Y.value] for x in outputs]]
            # mistery - why there are 1-element tuples?
            inputs_dates = [item[0] for item in [x[BatchKeys.DATES_PAST.value] for x in outputs]]
            labels_dates = [item[0] for item in [x[BatchKeys.DATES_FUTURE.value] for x in outputs]]
        output_series = np.asarray([np.asarray(el) for el in output_series])
        labels_series = np.asarray([np.asarray(el) for el in labels_series])
        gfs_targets = np.asarray([np.asarray(el) for el in gfs_targets])
        past_truth_series = np.asarray([np.asarray(el) for el in past_truth_series])

        gfs_corr = np.corrcoef(gfs_targets.flatten(), output_series.flatten())[0, 1]
        rmse_by_step = np.sqrt(np.mean(np.power(np.subtract(output_series, labels_series), 2), axis=0))

        # for plots
        plot_truth_series = []
        plot_prediction_series = []
        plot_gfs_targets = []
        plot_all_dates = []
        plot_prediction_dates = []
        for index in np.random.choice(np.arange(len(output_series)),
                                      min(40, len(output_series)), replace=False):
            sample_all_dates = [pd.to_datetime(pd.Timestamp(d)) for d in inputs_dates[index]]
            sample_prediction_dates = [pd.to_datetime(pd.Timestamp(d)) for d in labels_dates[index]]
            sample_all_dates.extend(sample_prediction_dates)

            plot_all_dates.append(sample_all_dates)
            plot_prediction_dates.append(sample_prediction_dates)

            plot_prediction_series.append(output_series[index])
            plot_truth_series.append(np.concatenate([past_truth_series[index], labels_series[index]], 0).tolist())
            plot_gfs_targets.append(gfs_targets[index])

        return {
            'epoch': float(step),
            'test_rmse': math.sqrt(float(self.test_mse.compute().item())),
            'test_mae': float(self.test_mae.compute().item()),
            'test_mase': float(self.test_mase.compute()),
            'gfs_corr': gfs_corr,
            'rmse_by_step': rmse_by_step,
            'plot_truth': plot_truth_series,
            'plot_prediction': plot_prediction_series,
            'plot_all_dates': plot_all_dates,
            'plot_prediction_dates': plot_prediction_dates,
            'plot_gfs_targets': plot_gfs_targets
        }

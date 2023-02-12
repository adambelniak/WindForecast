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

        if self.categorical_experiment:
            self.metrics_for_categorical_experiment(outputs, targets, past_targets, 'test')

            targets *= self.classes
            past_targets *= self.classes
        else:
            self.test_mse(outputs, targets)
            self.test_mae(outputs, targets)
            if len(targets.shape) == 1:
                self.test_mase(outputs.unsqueeze(0), targets.unsqueeze(0), past_targets.unsqueeze(0))
            else:
                self.test_mase(outputs, targets, past_targets)

        dates_inputs = batch[BatchKeys.DATES_PAST.value]
        dates_targets = batch[BatchKeys.DATES_FUTURE.value]

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value]

        return {BatchKeys.SYNOP_FUTURE_Y.value: batch[BatchKeys.SYNOP_FUTURE_Y.value].float().squeeze(),
                BatchKeys.PREDICTIONS.value: outputs.squeeze() if not self.categorical_experiment else torch.argmax(outputs, dim=-1),
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
        print(metrics_and_plot_results['test_mase'])

    def get_metrics_and_plot_results(self, step: int, outputs: List[Any]) -> Dict:
        series = self.get_series_from_outputs(outputs)
        predictions = series[BatchKeys.PREDICTIONS.value]

        if self.cfg.experiment.batch_size == 1:
            gfs_targets = [item.cpu() for item in [x[BatchKeys.GFS_FUTURE_Y.value] for x in outputs]]
        else:
            gfs_targets = [item.cpu() for sublist in [x[BatchKeys.GFS_FUTURE_Y.value] for x in outputs] for item in
                           sublist]
        output_series = np.asarray([np.asarray(el) for el in predictions])
        labels_series = np.asarray([np.asarray(el) for el in series[BatchKeys.SYNOP_FUTURE_Y.value]])
        gfs_targets = np.asarray([np.asarray(el) for el in gfs_targets])
        past_truth_series = np.asarray([np.asarray(el) for el in series[BatchKeys.SYNOP_PAST_Y.value]])

        gfs_corrs = []
        for index, _ in enumerate(gfs_targets):
            gfs_corrs.append(np.corrcoef(gfs_targets[index], output_series[index])[0, 1])

        gfs_corr = np.mean(gfs_corrs)
        rmse_by_step = np.sqrt(np.mean(np.power(np.subtract(output_series, labels_series), 2), axis=0))
        mase_by_step = self.get_mase_by_step(predictions, series[BatchKeys.SYNOP_FUTURE_Y.value], series[BatchKeys.SYNOP_PAST_Y.value])

        # for plots
        plot_truth_series = []
        plot_prediction_series = []
        plot_gfs_targets = []
        plot_all_dates = []
        plot_prediction_dates = []
        for index in np.random.choice(np.arange(len(output_series)),
                                      min(40, len(output_series)), replace=False):
            sample_all_dates = [pd.to_datetime(pd.Timestamp(d)) for d in series[BatchKeys.DATES_PAST.value][index]]
            sample_prediction_dates = [pd.to_datetime(pd.Timestamp(d)) for d in series[BatchKeys.DATES_FUTURE.value][index]]
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
            'mase_by_step': mase_by_step,
            'plot_truth': plot_truth_series,
            'plot_prediction': plot_prediction_series,
            'plot_all_dates': plot_all_dates,
            'plot_prediction_dates': plot_prediction_dates,
            'plot_gfs_targets': plot_gfs_targets,
            'output_series': output_series,
            'gfs_targets': gfs_targets,
            'truth_series': labels_series
        }

import wandb

from wind_forecast.config.register import Config
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(system, config: Config, mean, std, min, max):
    for index in np.random.choice(np.arange(len(system.test_results['output'])), min(40, len(system.test_results['output'])), replace=False):
        fig, ax = plt.subplots()
        inputs_dates = [pd.to_datetime(pd.Timestamp(d)) for d in system.test_results['inputs_dates'][index]]
        output_dates = [pd.to_datetime(pd.Timestamp(d)) for d in system.test_results['targets_dates'][index]]

        inputs_dates.extend(output_dates)
        x = inputs_dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())

        out_series = system.test_results['output'][index].cpu()
        truth_series = system.test_results['inputs'][index].cpu()
        truth_series.extend(system.test_results['labels'][index].cpu())
        if config.experiment.use_gfs_data:
            gfs_out_series = system.test_results['gfs_targets'][index].cpu()

        if mean is not None and std is not None:
            # TODO think about a more robust solution
            if type(mean) == list:
                mean = mean[0]
            if type(std) == list:
                mean = std[0]
            out_series = out_series * std + mean
            truth_series = (truth_series * std + mean).tolist()
            if config.experiment.use_gfs_data:
                gfs_out_series = gfs_out_series * std + mean
        elif min is not None and max is not None:
            if type(min) == list:
                min = min[0]
            if type(max) == list:
                max = max[0]
            out_series = out_series * (max - min) + min
            truth_series = (truth_series * (max - min) + min).tolist()
            if config.experiment.use_gfs_data:
                gfs_out_series = gfs_out_series * (max - min) + min

        ax.plot(output_dates, out_series, label='prediction')

        if config.experiment.use_gfs_data:
            ax.plot(output_dates, gfs_out_series, label='gfs prediction')

        ax.plot(x, truth_series, label='ground truth')
        ax.set_xlabel('Date')
        ax.set_ylabel(config.experiment.target_parameter)
        ax.legend(loc='best')
        plt.gcf().autofmt_xdate()
        wandb.log({'chart': ax})

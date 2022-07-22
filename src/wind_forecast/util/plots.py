import wandb

from wind_forecast.config.register import Config
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
K_TO_C = 273.15

def plot_results(system, config: Config, mean, std, gfs_mean, gfs_std):
    for index in np.random.choice(np.arange(len(system.test_results['output'])), min(40, len(system.test_results['output'])), replace=False):
        fig, ax = plt.subplots()
        inputs_dates = [pd.to_datetime(pd.Timestamp(d)) for d in system.test_results['inputs_dates'][index]]
        output_dates = [pd.to_datetime(pd.Timestamp(d)) for d in system.test_results['targets_dates'][index]]

        inputs_dates.extend(output_dates)
        x = inputs_dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())

        out_series = system.test_results['output'][index].cpu().tolist()
        truth_series = system.test_results['inputs'][index].cpu().tolist()
        truth_series.extend(system.test_results['labels'][index].cpu().tolist())
        if config.experiment.use_gfs_data:
            gfs_out_series = system.test_results['gfs_targets'][index].cpu().tolist()

        if mean is not None and std is not None:
            # TODO think about a more robust solution
            if type(mean) == list:
                mean = mean[0]
            if type(std) == list:
                std = std[0]
            truth_series = (np.array(truth_series) * std + mean).tolist()
            if config.experiment.use_gfs_data:
                gfs_out_series = np.array(gfs_out_series) * gfs_std + gfs_mean - K_TO_C
                if config.experiment.differential_forecast:
                    out_series = (np.array(out_series) * std + gfs_out_series).tolist()
                else:
                    out_series = (np.array(out_series) * std + mean).tolist()
                gfs_out_series = gfs_out_series.tolist()

        ax.plot(output_dates, out_series, label='prediction')

        if config.experiment.use_gfs_data:
            ax.plot(output_dates, gfs_out_series, label='gfs prediction')

        ax.plot(x, truth_series, label='ground truth')
        ax.set_xlabel('Date')
        ax.set_ylabel(config.experiment.target_parameter)
        ax.legend(loc='best')
        plt.gcf().autofmt_xdate()
        wandb.log({'chart': ax})

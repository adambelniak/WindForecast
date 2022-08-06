import wandb
from wind_forecast.config.register import Config
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(system, config: Config, mean, std, gfs_mean, gfs_std):
    plot_random_series(system, config, mean, std, gfs_mean, gfs_std)
    plot_step_by_step_metric(system)


def plot_random_series(system, config: Config, mean, std, gfs_mean, gfs_std):
    K_TO_C = 273.15 if config.experiment.target_parameter == 'temperature' else 0
    for index in range(len(system.test_results['plot_truth'])):
        fig, ax = plt.subplots()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())

        prediction_series = system.test_results['plot_prediction'][index]
        truth_series = system.test_results['plot_truth'][index]
        prediction_dates = system.test_results['plot_prediction_dates'][index]
        truth_dates = system.test_results['plot_truth_dates'][index]
        if config.experiment.use_gfs_data:
            gfs_out_series = system.test_results['plot_gfs_targets'][index]

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
                    prediction_series = (np.array(prediction_series) * std + gfs_out_series).tolist()
                else:
                    prediction_series = (np.array(prediction_series) * std + mean).tolist()
                gfs_out_series = gfs_out_series.tolist()

        ax.plot(prediction_dates, prediction_series, label='prediction')

        if config.experiment.use_gfs_data:
            ax.plot(prediction_dates, gfs_out_series, label='gfs prediction')

        ax.plot(truth_dates, truth_series, label='ground truth')
        ax.set_xlabel('Date')
        ax.set_ylabel(config.experiment.target_parameter)
        ax.legend(loc='best')
        plt.gcf().autofmt_xdate()
        wandb.log({'series_chart': ax})
        plt.close(fig)

def plot_step_by_step_metric(system):
    rmse_by_step = system.test_results['rmse_by_step']

    plt.plot(np.arange(rmse_by_step.shape[0]), rmse_by_step, '-x')
    plt.xlabel('step')
    plt.ylabel('rmse')
    wandb.log({'step_by_step_metric_chart': plt.gca()})
    plt.close()


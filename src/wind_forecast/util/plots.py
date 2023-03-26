import os

import wandb

from synop.consts import LOWER_CLOUDS, CLOUD_COVER
from wind_forecast.config.register import Config
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from wind_forecast.util.gfs_util import get_gfs_target_param


def plot_results(system, config: Config, synop_mean, synop_std, gfs_mean, gfs_std):
    plot_random_series(system, config, synop_mean, synop_std, gfs_mean, gfs_std)
    plot_step_by_step_metric(system)
    plot_scatters(system, config, synop_mean, synop_std, gfs_mean, gfs_std)


def plot_predict(system, config: Config, synop_mean, synop_std, gfs_mean, gfs_std):
    fig, ax = plt.subplots()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())

    prediction_series = system.predict_results['prediction']
    truth_series = system.predict_results['truth_series']
    prediction_dates = system.predict_results['prediction_dates']
    all_dates = system.predict_results['all_dates']

    past_dates = system.predict_results['past_dates']
    truth_series = rescale_series(config, truth_series, synop_mean, synop_std, config.experiment.target_parameter)
    prediction_series = rescale_series(config, prediction_series, synop_mean, synop_std, config.experiment.target_parameter)

    ax.plot(prediction_dates, prediction_series, label='prediction')
    if config.experiment.use_gfs_data:
        gfs_out_series = system.predict_results['gfs_targets']
        gfs_out_series = rescale_series(config, gfs_out_series, gfs_mean, gfs_std, get_gfs_target_param(config.experiment.target_parameter))
        gfs_past_series = system.predict_results['gfs_past']
        gfs_past_series = rescale_series(config, gfs_past_series, gfs_mean, gfs_std, get_gfs_target_param(config.experiment.target_parameter))
        ax.plot(all_dates, np.concatenate([gfs_past_series, gfs_out_series]), label='gfs prediction')

    ax.plot(past_dates, truth_series, label='ground truth')

    ax.set_xlabel('Date')
    ax.set_ylabel(config.experiment.target_parameter)
    ax.legend(loc='best')
    plt.gcf().autofmt_xdate()
    wandb.log({'series_chart': ax})
    plt.close(fig)


def rescale_series(config: Config, series, mean, std, param):
    if config.experiment.target_parameter in [LOWER_CLOUDS[1], CLOUD_COVER[1]]:
        # These series are already properly scaled
        return series
    if mean is not None and std is not None:
        # TODO think about a more robust solution
        if type(mean) == list:
            mean = mean[0]
        if type(std) == list:
            std = std[0]
        if type(mean) == dict:
            mean = mean[param]
        if type(std) == dict:
            std = std[param]

        ret = np.array(series) * std + mean
        if param == 'TMP_HTGL_2':
            return ret - 273.15
        if param == 'PRES_SFC_0':
            return ret / 100
        return ret

    return series


def plot_random_series(system, config: Config, synop_mean, synop_std, gfs_mean, gfs_std):
    for index in range(len(system.test_results['plot_truth'])):
        fig, ax = plt.subplots()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())

        prediction_series = system.test_results['plot_prediction'][index]
        truth_series = system.test_results['plot_truth'][index]
        prediction_dates = system.test_results['plot_prediction_dates'][index]
        all_dates = system.test_results['plot_all_dates'][index]

        truth_series = rescale_series(config, truth_series, synop_mean, synop_std, config.experiment.target_parameter)
        prediction_series = rescale_series(config, prediction_series, synop_mean, synop_std, config.experiment.target_parameter)

        if config.experiment.use_gfs_data:
            gfs_out_series = system.test_results['plot_gfs_targets'][index]
            gfs_out_series = rescale_series(config, gfs_out_series, gfs_mean, gfs_std, get_gfs_target_param(config.experiment.target_parameter))

        ax.plot(prediction_dates, prediction_series, label='prediction')

        if config.experiment.use_gfs_data:
            ax.plot(prediction_dates, gfs_out_series, label='gfs prediction')
        ax.plot(all_dates, truth_series, label='ground truth')
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


def plot_scatters(system, config: Config, synop_mean, synop_std, gfs_mean, gfs_std):
    output_series = system.test_results['output_series']
    truth_series = system.test_results['truth_series']
    output_series = rescale_series(config, output_series, synop_mean, synop_std, config.experiment.target_parameter)
    truth_series = rescale_series(config, truth_series, synop_mean, synop_std, config.experiment.target_parameter)

    if config.experiment.use_gfs_data:
        gfs_targets = system.test_results['gfs_targets']
        gfs_targets = rescale_series(config, gfs_targets, gfs_mean, gfs_std, get_gfs_target_param(config.experiment.target_parameter))

        plot_scatter(output_series, gfs_targets, "Predykcja modelu", "Predykcja GFS", "scatter_plot_gfs_pred.png")
        plot_scatter(truth_series, gfs_targets, "Wartość rzeczywista", "Predykcja GFS", "scatter_plot_gfs_truth.png")

    plot_scatter(truth_series, output_series, "Wartość rzeczywista", "Predykcja modelu", "scatter_plot_pred_truth.png")


def plot_scatter(series_a, series_b, label_a, label_b, filename):
    fig, ax = plt.subplots(figsize=(30, 15), dpi=80)
    ax.scatter([x for series in series_a for x in series], [x for series in series_b for x in series], s=1)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, color='black')
    ax.set_xlabel(label_a, fontsize = 18)
    ax.set_ylabel(label_b, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{filename}')
    plt.close()

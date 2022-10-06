import itertools
import os
from typing import Any, List, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import wandb
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.util.config import process_config
from datetime import datetime


def run_analysis(config: Config):
    analysis_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 'config', 'analysis',
                                 config.analysis.input_file)
    analysis_config = process_config(analysis_file)
    entity = os.getenv('WANDB_ENTITY', '')
    project = os.getenv('WANDB_PROJECT', '')
    run_summaries = []
    run_configs = []
    for run in analysis_config.runs:
        run_id = run['id']
        api = wandb.Api()
        wandb_run = api.run(f"{entity}/{project}/{run_id}")
        run_summaries.append(wandb_run.summary)
        run_configs.append(wandb_run.config)

    plot_series_comparison(analysis_config.runs, run_summaries, run_configs)
    plot_rmse_by_step_comparison(analysis_config.runs, run_summaries)


def plot_series_comparison(analysis_config_runs: List, run_summaries: List[Any], run_configs: List[Dict]):
    first_run_configs = run_configs[0]

    truth_series = run_summaries[0]['plot_truth']
    all_dates = run_summaries[0]['plot_all_dates']
    prediction_dates = run_summaries[0]['plot_prediction_dates']
    target_mean = run_summaries[0]['target_mean_0']
    target_std = run_summaries[0]['target_std_0']
    for series_index in range(len(truth_series)):
        fig, ax = plt.subplots(figsize=(30, 15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

        ax.plot([datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in all_dates[series_index]],
                     (np.array(truth_series[series_index]) * target_std + target_mean).tolist(), label='ground truth')

        for index, run in enumerate(run_summaries):
            prediction_series = run['plot_prediction'][series_index]
            prediction_series = (np.array(prediction_series) * target_std + target_mean).tolist()
            ax.plot([datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in prediction_dates[series_index]],
                         prediction_series, label=analysis_config_runs[index]['axis_label'])

        target_param = first_run_configs['experiment/target_parameter']
        # Labels hardcoded for now
        ax.set_ylabel("Temperatura" if target_param == 'temperature' else "Prędkość wiatru", fontsize=18)
        ax.set_xlabel('Data', fontsize=18)
        ax.legend(loc='best', prop={'size': 18})
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.gcf().autofmt_xdate()
        os.makedirs('analysis', exist_ok=True)
        plt.savefig(f'analysis/series_comparison_{series_index}.png')
        plt.close()


def plot_rmse_by_step_comparison(analysis_config_runs: List, run_summaries: List[Any]):
    fig, ax = plt.subplots(figsize=(30, 15))
    marker = itertools.cycle((',', '+', '.', 'o', '*', 'x'))

    for index, run in enumerate(run_summaries):
        rmse_by_step = run['rmse_by_step']

        ax.plot(np.arange(len(rmse_by_step)), rmse_by_step, marker=next(marker), linestyle='solid',
                 label=analysis_config_runs[index]['axis_label'])

    ax.set_ylabel('RMSE', fontsize=18)
    ax.set_xlabel('Krok', fontsize=18)
    ax.legend(loc='best', prop={'size': 18})
    ax.tick_params(axis='both', which='major', labelsize=14)
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(f'analysis/rmse_by_step_comparison.png')
    plt.close()

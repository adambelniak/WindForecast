import itertools
import os
from typing import Any, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import wandb
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.util.config import process_config
from datetime import datetime

marker = itertools.cycle(('+', '.', 'o', '*', 'x', 'v', 'D'))

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

    plot_series_comparison(analysis_config.runs, run_summaries, config)
    plot_rmse_by_step_comparison(analysis_config.runs, run_summaries)
    plot_mase_by_step_comparison(analysis_config.runs, run_summaries)
    plot_gfs_corr_comparison()


def plot_series_comparison(analysis_config_runs: List, run_summaries: List[Any], config: Config):
    truth_series = run_summaries[0]['plot_truth']
    all_dates = run_summaries[0]['plot_all_dates']
    prediction_dates = run_summaries[0]['plot_prediction_dates']
    target_mean = run_summaries[0]['target_mean_0'] if 'target_mean_0' in run_summaries[0].keys() else run_summaries[0]['target_mean']
    target_std = run_summaries[0]['target_std_0'] if 'target_std_0' in run_summaries[0].keys() else run_summaries[0]['target_std']

    for series_index in range(len(truth_series)):
        fig, ax = plt.subplots(figsize=(30, 15))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

        ax.plot([datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in all_dates[series_index]],
                     (np.array(truth_series[series_index]) * target_std + target_mean).tolist(), label='Wartość rzeczywista', linewidth=4)

        for index, run in enumerate(run_summaries):
            prediction_series = run['plot_prediction'][series_index]
            prediction_series = (np.array(prediction_series) * target_std + target_mean).tolist()
            ax.plot([datetime.strptime(date, '%Y-%m-%dT%H:%M:%S') for date in prediction_dates[series_index]],
                         prediction_series, label=analysis_config_runs[index]['axis_label'], marker=next(marker))

        # Labels hardcoded for now
        ax.set_ylabel(config.analysis.target_parameter, fontsize=20)
        ax.set_xlabel('Data', fontsize=20)
        ax.legend(loc='best', prop={'size': 18})
        ax.tick_params(axis='both', which='major', labelsize=14)
        plt.gcf().autofmt_xdate()
        os.makedirs('analysis', exist_ok=True)
        plt.savefig(f'analysis/series_comparison_{series_index}.png')
        plt.close()


def plot_rmse_by_step_comparison(analysis_config_runs: List, run_summaries: List[Any]):
    fig, ax = plt.subplots(figsize=(30, 15))

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

def plot_mase_by_step_comparison(analysis_config_runs: List, run_summaries: List[Any]):
    fig, ax = plt.subplots(figsize=(30, 15))

    for index, run in enumerate(run_summaries):
        mase_by_step = run['mase_by_step']

        ax.plot(np.arange(len(mase_by_step)), mase_by_step, marker=next(marker), linestyle='solid',
                 label=analysis_config_runs[index]['axis_label'])

    ax.set_ylabel('MASE', fontsize=18)
    ax.set_xlabel('Krok', fontsize=18)
    ax.legend(loc='best', prop={'size': 18})
    ax.tick_params(axis='both', which='major', labelsize=14)
    os.makedirs('analysis', exist_ok=True)
    plt.savefig(f'analysis/mase_by_step_comparison.png')
    plt.close()

def plot_gfs_corr_comparison():
    # for now hardcoded
    labels = ['N-BEATSx + GFS', 'LSTM + GFS', 'BiLSTM + GFS', "TCN + GFS", "TCNAttention + GFS", "Transformer + GFS",
              "Spacetimeformer + GFS", "Regresja liniowa"]

    temp_corrs = [0.8212, 0.8231, 0.8217, 0.8451, 0.8692, 0.8334, 0.9912, 0.5514]
    wind_corrs = [0.5395, 0.5048, 0.4821, 0.4943, 0.5669, 0.5898, 0.8279, 0.3707]
    pres_corrs = [0.868, 0.8823, 0.8678, 0.8709, 0.8533, 0.8662, 0.9488, 0.171]
    x = np.arange(len(labels))
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(25, 10))
    rects1 = ax.bar(x - width, temp_corrs, width, label='Temperatura')
    rects2 = ax.bar(x, wind_corrs, width, label='Prędkość wiatru')
    rects3 = ax.bar(x + width, pres_corrs, width, label='Ciśnienie')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Korelacja', fontsize=20)
    plt.tick_params(labelsize=14)
    plt.xticks(x, labels)

    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    os.makedirs('analysis', exist_ok=True)
    plt.savefig(f'analysis/gfs_corr.png')
    plt.close()


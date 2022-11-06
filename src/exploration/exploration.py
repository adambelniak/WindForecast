import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

from gfs_archive_0_25.gfs_processor.own_logger import get_logger
from gfs_archive_0_25.utils import prep_zeros_if_needed
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from synop.consts import SYNOP_TRAIN_FEATURES
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.gfs_util import GFS_DATASET_DIR, get_available_numpy_files

GFS_PARAMETERS = [
    {
        "name": "T CDC",
        "level": "LCY_0"
    },
    {
        "name": "T CDC",
        "level": "MCY_0"
    },
    {
        "name": "T CDC",
        "level": "HCY_0"
    },
    {
        "name": "V GRD",
        "level": "HTGL_10"
    },
    {
        "name": "U GRD",
        "level": "HTGL_10"
    },
    {
        "name": "TMP",
        "level": "ISBL_850"
    },
    {
        "name": "TMP",
        "level": "HTGL_2"
    },
    {
        "name": "PRATE",
        "level": "SFC_0"
    },
    {
        "name": "PRES",
        "level": "SFC_0"
    },
    {
        "name": "HGT",
        "level": "ISBL_850"
    },
    {
      "name": "HGT",
      "level": "ISBL_500"
    },
    {
        "name": "R H",
        "level": "HTGL_2"
    },
    {
        "name": "R H",
        "level": "ISBL_900"
    },
    {
        "name": "R H",
        "level": "ISBL_800"
    },
    {
        "name": "R H",
        "level": "ISBL_700"
    },
    {
        "name": "R H",
        "level": "ISBL_500"
    },
    {
        "name": "DPT",
        "level": "HTGL_2"
    },
    {
        "name": "CAPE",
        "level": "SFC_0"
    },
]
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def explore_data_for_each_gfs_param():
    logger = get_logger(os.path.join("explore_results", 'logs.log'))
    for parameter in tqdm(GFS_PARAMETERS):
        param_dir = os.path.join(GFS_DATASET_DIR, parameter['name'], parameter['level'])
        if os.path.exists(param_dir):
            for offset in tqdm(range(3, 120, 3)):
                plot_dir = os.path.join('plots', parameter['name'], parameter['level'], str(offset))
                if not os.path.exists(os.path.join(plot_dir, 'plot.png')):
                    matcher = re.compile(rf".*f{prep_zeros_if_needed(str(offset), 2)}.*")
                    files = [f.name for f in os.scandir(param_dir) if matcher.match(f.name)]
                    files_with_nan = []
                    values = np.empty((len(files), 33, 53))
                    for index, file in enumerate(files):
                        values[index] = np.load(os.path.join(param_dir, file))
                        if np.isnan(np.sum(values[index])):
                            files_with_nan.append(file)

                    values = values.flatten()
                    sns.boxplot(x=values).set_title(f"{parameter['name']} {parameter['level']}")
                    os.makedirs(plot_dir, exist_ok=True)
                    plt.savefig(os.path.join(plot_dir, 'plot.png'))
                    plt.close()
                    if len(files_with_nan) > 0:
                        logger.info(
                            f"Nans in files:\n {[f'{os.path.join(param_dir, file.name)}, ' for file in files_with_nan]}")


def explore_gfs_correlations():
    if os.path.exists(os.path.join(Path(__file__).parent, "gfs_heatmap.png")):
        return

    offset = 3
    df = pd.DataFrame()
    files = get_available_numpy_files(GFS_PARAMETERS, offset)
    for parameter in tqdm(GFS_PARAMETERS):
        param_dir = os.path.join(GFS_DATASET_DIR, parameter['name'], parameter['level'])
        values = np.empty((len(files), 33, 53))
        for index, file in tqdm(enumerate(files)):
            values[index] = np.load(os.path.join(param_dir, file))
        df[parameter['name'] + "_" + parameter['level']] = values.flatten()

    print(f"Number of files: {len(files)}")
    sns.heatmap(df.corr(), annot=True)
    plt.savefig(os.path.join(Path(__file__).parent, "gfs_heatmap.png"))
    plt.close()


def explore_synop_correlations(data: pd.DataFrame, features: (int, str), localisation_name: str):
    if os.path.exists(os.path.join(Path(__file__).parent, f"synop_{localisation_name}_heatmap.png")):
        return
    data = data[list(list(zip(*features))[2])]
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(), annot=True, annot_kws={"fontsize":12})
    plt.savefig(os.path.join(Path(__file__).parent, f"synop_{localisation_name}_heatmap.png"), dpi=200)
    plt.close()


def explore_synop_patterns(data: pd.DataFrame, features: (int, str), localisation_name: str):
    features_with_nans = []
    for feature in features:
        plot_dir = os.path.join('plots-synop', localisation_name, feature[1])
        values = data[feature[2]].to_numpy()
        if np.isnan(np.sum(values)):
            features_with_nans.append(feature[2])
        sns.boxplot(x=values).set_title(f"{feature[2]}")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'plot-box.png'))
        plt.close()

        stationarity_test = adfuller(values)
        print(f"Stationarity test for {feature[1]}")
        print('ADF Statistic: %f' % stationarity_test[0])
        print('p-value: %f' % stationarity_test[1])
        print('Critical Values:')
        for key, value in stationarity_test[4].items():
            print('\t%s: %.3f' % (key, value))

        _, ax = plt.subplots(figsize=(30, 15))
        ax.set_xlabel('Data', fontsize=22)
        ax.set_ylabel(feature[2], fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        sns.lineplot(ax=ax, data=data[['date', feature[2]]], x='date', y=feature[2])
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'plot-line.png'))
        plt.close()

        data2 = data.iloc[2000:2600]
        _, ax = plt.subplots(figsize=(30, 15))
        ax.set_xlabel('Data', fontsize=22)
        ax.set_ylabel(feature[2], fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        sns.lineplot(ax=ax, data=data2[['date', feature[2]]], x='date', y=feature[2])
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'plot-line2.png'))
        plt.close()

    if len(features_with_nans):
        logger = get_logger(os.path.join("explore_results", 'logs.log'))
        logger.info(f"Nans in features:\n {[f'{feature}, ' for feature in features_with_nans]}")


def explore_synop(synop_file):
    relevant_features = [f for f in SYNOP_TRAIN_FEATURES if f[1] not in ['year', 'month', 'day', 'hour']]
    if not os.path.exists(synop_file):
        raise Exception(f"CSV file with synop data does not exist at path {synop_file}.")

    data = prepare_synop_dataset(synop_file, list(list(zip(*relevant_features))[1]), norm=False,
                                 dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=2016, to_year=2022, decompose_periodic=False)

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data = data.rename(columns=dict(zip([f[1] for f in SYNOP_TRAIN_FEATURES], [f[2] for f in SYNOP_TRAIN_FEATURES])))
    explore_synop_correlations(data, relevant_features, os.path.basename(synop_file))
    explore_synop_patterns(data, relevant_features, os.path.basename(synop_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--synop_csv', help='Path to a CSV file with synop data',
                        default=os.path.join(SYNOP_DATASETS_DIRECTORY, 'WARSZAWA-OKECIE_352200375_data.csv'), type=str)
    parser.add_argument('--skip_gfs', help='Skip GFS dataset.', action='store_true')


    args = parser.parse_args()

    if not args.skip_gfs:
        explore_data_for_each_gfs_param()
        explore_gfs_correlations()
    explore_synop(args.synop_csv)

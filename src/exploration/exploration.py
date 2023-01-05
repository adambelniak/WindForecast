import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

from gfs_archive_0_25.gfs_processor.Coords import Coords
from gfs_archive_0_25.gfs_processor.own_logger import get_logger
from gfs_archive_0_25.utils import prep_zeros_if_needed
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from synop.consts import SYNOP_TRAIN_FEATURES, TEMPERATURE, VELOCITY_COLUMN, PRESSURE, DIRECTION_COLUMN
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.df_util import resolve_indices
from wind_forecast.util.gfs_util import GFS_DATASET_DIR, get_available_numpy_files, GFSUtil, get_gfs_target_param, \
    extend_wind_components
from wind_forecast.util.synop_util import get_correct_dates_for_sequence

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


GFS_PARAMS_CONFIG = {
  "params": [
    {
      "name": "T CDC",
      "level": "LCY_0",
      "interpolation": "polynomial",
      "min": 0,
      "max": 100
    },
    {
      "name": "T CDC",
      "level": "MCY_0",
      "interpolation": "polynomial",
      "min": 0,
      "max": 100
    },
    {
      "name": "T CDC",
      "level": "HCY_0",
      "interpolation": "polynomial",
      "min": 0,
      "max": 100
    },
    {
      "name": "V GRD",
      "level": "HTGL_10",
      "interpolation": "polynomial"
    },
    {
      "name": "U GRD",
      "level": "HTGL_10",
      "interpolation": "polynomial"
    },
    {
      "name": "TMP",
      "level": "ISBL_850",
      "interpolation": "polynomial"
    },
    {
      "name": "TMP",
      "level": "HTGL_2",
      "interpolation": "polynomial"
    },
    {
      "name": "PRATE",
      "level": "SFC_0",
      "interpolation": "polynomial",
      "min": 0
    },
    {
      "name": "PRES",
      "level": "SFC_0",
      "interpolation": "polynomial"
    },
    {
      "name": "R H",
      "level": "HTGL_2",
      "interpolation": "polynomial",
      "min": 0,
      "max": 100
    },
    {
      "name": "R H",
      "level": "ISBL_700",
      "interpolation": "polynomial",
      "min": 0,
      "max": 100
    },
    {
      "name": "DPT",
      "level": "HTGL_2",
      "interpolation": "polynomial"
    },
        {
      "name": "HGT",
      "level": "ISBL_500",
      "interpolation": "polynomial"
    },
        {
      "name": "TMP",
      "level": "ISBL_700",
      "interpolation": "polynomial"
    },
    {
      "name": "GUST",
      "level": "SFC_0",
      "interpolation": "polynomial",
      "min": 0
    },
    {
        "name": "ALBDO",
        "level": "SFC_0",
        "interpolation": "polynomial",
        "min": 0
    },
    {
        "name": "SNO D",
        "level": "SFC_0",
        "interpolation": "polynomial",
        "min": 0
    }
  ]
}


def get_synop_data(synop_filepath: str):
    if not os.path.exists(synop_filepath):
        raise Exception(f"CSV file with synop data does not exist at path {synop_filepath}.")

    data = prepare_synop_dataset(synop_filepath, list(list(zip(*SYNOP_TRAIN_FEATURES))[1]), norm=False,
                                 dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=2016, to_year=2022, decompose_periodic=False)

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    return data.rename(columns=dict(zip([f[1] for f in SYNOP_TRAIN_FEATURES], [f[2] for f in SYNOP_TRAIN_FEATURES])))


def get_gfs_data_for_offset(offset=3):
    df = pd.DataFrame()
    files = get_available_numpy_files(GFS_PARAMETERS, offset)
    for parameter in tqdm(GFS_PARAMETERS):
        param_dir = os.path.join(GFS_DATASET_DIR, parameter['name'], parameter['level'])
        values = np.empty((len(files), 33, 53))
        for index, file in tqdm(enumerate(files)):
            values[index] = np.load(os.path.join(param_dir, file))
        df[parameter['name'] + "_" + parameter['level']] = values.flatten()

    print(f"Number of files: {len(files)}")
    return df


def prepare_gfs_data_with_wind_components(gfs_data: pd.DataFrame):
    gfs_wind_parameters = ["V GRD_HTGL_10", "U GRD_HTGL_10"]
    gfs_wind_data = gfs_data[gfs_wind_parameters]
    gfs_data.drop(columns=gfs_wind_parameters, inplace=True)
    velocity, sin, cos = extend_wind_components(gfs_wind_data.values)
    gfs_data["wind-velocity"] = velocity
    gfs_data["wind-sin"] = sin
    gfs_data["wind-cos"] = cos
    return gfs_data


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

    df = get_gfs_data_for_offset()

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


def plot_diff_hist(diff, parameter: str):
    plt.figure(figsize=(20, 10))
    plt.tight_layout()
    sns.displot(diff, bins=100, kde=True)
    plt.ylabel('Liczebność')
    plt.xlabel('Różnica')
    plt.title('Prognoza vs obserwacja, ' + parameter)

    os.makedirs(os.path.join(Path(__file__).parent, "plots"), exist_ok=True)
    plt.savefig(os.path.join(Path(__file__).parent, "plots", f"gfs_diff_{parameter}.png"), dpi=200, bbox_inches='tight')


def plot_diff_line(x, y, xlabel, ylabel, parameter: str):
    plt.figure(figsize=(20, 10))
    plt.tight_layout()
    sns.lineplot(x=x, y=y)
    plt.plot([0, 360], [0, 0], linewidth=2, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Błąd prognozy vs kierunek wiatru, " + parameter.lower())

    os.makedirs(os.path.join(Path(__file__).parent, "plots"), exist_ok=True)
    plt.savefig(os.path.join(Path(__file__).parent, "plots", f"gfs_diff_{parameter}_by_direction.png"), dpi=200,
                bbox_inches='tight')


def explore_gfs_bias(synop_file: str, target_coords: tuple):
    synop_data = get_synop_data(synop_file)
    synop_dates = get_correct_dates_for_sequence(synop_data, 24, 24, 0)
    synop_data = synop_data.reset_index()
    data_indices = synop_data[synop_data["date"].isin(synop_dates)].index

    gfs_util = GFSUtil(Coords(target_coords[0], target_coords[0], target_coords[1], target_coords[1]),
                       24, 24, 0, GFS_PARAMS_CONFIG['params'])
    data_indices, gfs_data = gfs_util.match_gfs_with_synop_sequence2sequence(synop_data, data_indices)
    all_synop_data = resolve_indices(synop_data, data_indices, 48)
    gfs_data = prepare_gfs_data_with_wind_components(gfs_data)
    all_gfs_data = resolve_indices(gfs_data, data_indices, 48)

    gfs_targets = all_gfs_data[get_gfs_target_param(TEMPERATURE[1])].values
    synop_targets = all_synop_data[TEMPERATURE[2]].values
    gfs_targets -= 273.15 # convert K to C

    diff = gfs_targets - synop_targets
    plot_diff_hist(diff, TEMPERATURE[2])

    gfs_targets = all_gfs_data[get_gfs_target_param(VELOCITY_COLUMN[1])].values
    synop_targets = all_synop_data[VELOCITY_COLUMN[2]].values

    diff = gfs_targets - synop_targets
    plot_diff_hist(diff, VELOCITY_COLUMN[2])

    gfs_targets = all_gfs_data[get_gfs_target_param(PRESSURE[1])].values
    gfs_targets /= 100
    synop_targets = all_synop_data[PRESSURE[2]].values
    diff = gfs_targets - synop_targets
    plot_diff_hist(diff, PRESSURE[2])

    synop_wind = all_synop_data[[VELOCITY_COLUMN[2], DIRECTION_COLUMN[2]]]
    gfs_targets = all_gfs_data[get_gfs_target_param(VELOCITY_COLUMN[1])].values

    plot_diff_line(x=synop_wind[DIRECTION_COLUMN[2]],
                   y=gfs_targets - synop_wind[VELOCITY_COLUMN[2]],
                   xlabel='Kierunek wiatru, °',
                   ylabel='Różnica, m/s',
                   parameter=VELOCITY_COLUMN[2])

    synop_temp = all_synop_data[[TEMPERATURE[2], DIRECTION_COLUMN[2]]]
    gfs_targets = all_gfs_data[get_gfs_target_param(TEMPERATURE[1])].values

    plot_diff_line(x=synop_temp[DIRECTION_COLUMN[2]],
                   y=gfs_targets - synop_temp[TEMPERATURE[2]],
                   xlabel='Kierunek wiatru, °',
                   ylabel='Różnica, K',
                   parameter=TEMPERATURE[2])


def explore_synop(synop_file: str):
    data = get_synop_data(synop_file)
    explore_synop_correlations(data, SYNOP_TRAIN_FEATURES, os.path.basename(synop_file))
    explore_synop_patterns(data, SYNOP_TRAIN_FEATURES, os.path.basename(synop_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-synop_csv', help='Path to a CSV file with synop data',
                        default=os.path.join(SYNOP_DATASETS_DIRECTORY, 'WARSZAWA-OKECIE_352200375_data.csv'), type=str)
    parser.add_argument('--skip_gfs', help='Skip GFS dataset.', action='store_true')
    parser.add_argument('-target_coords', help='Coordinates of the target station.', default=(52.1831174, 20.9875259), type=tuple)


    args = parser.parse_args()

    if not args.skip_gfs:
        explore_data_for_each_gfs_param()
        explore_gfs_correlations()
        explore_gfs_bias(args.synop_csv, args.target_coords)
    explore_synop(args.synop_csv)

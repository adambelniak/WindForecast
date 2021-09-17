import argparse
import os
import re
from pathlib import Path

from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from gfs_archive_0_25.gfs_processor.own_logger import get_logger
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.consts import SYNOP_FEATURES
from wind_forecast.preprocess.synop.fetch_synop_data import download_list_of_station, get_localisation_id, \
    process_all_data
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.gfs_util import get_available_numpy_files
from gfs_archive_0_25.utils import prep_zeros_if_needed

gfs_dataset_dir = os.path.join("D:\\WindForecast", "output_np2")
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
    # {
    #   "name": "HGT",
    #   "level": "ISBL_500"
    # },
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


def explore_data_for_each_param():
    logger = get_logger(os.path.join("explore_results", 'logs.log'))
    for parameter in tqdm(GFS_PARAMETERS):
        param_dir = os.path.join(gfs_dataset_dir, parameter['name'], parameter['level'])
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
                        logger.info(f"Nans in files:\n {[f'{os.path.join(param_dir, file.name)}, ' for file in files_with_nan]}")


def explore_gfs_correlations():
    if os.path.exists(os.path.join(Path(__file__).parent, "gfs_heatmap.png")):
        return

    offset = 3
    df = pd.DataFrame()
    files = get_available_numpy_files(GFS_PARAMETERS, offset, gfs_dataset_dir)
    for parameter in tqdm(GFS_PARAMETERS):
        param_dir = os.path.join(gfs_dataset_dir, parameter['name'], parameter['level'])
        values = np.empty((len(files), 33, 53))
        for index, file in tqdm(enumerate(files)):
            values[index] = np.load(os.path.join(param_dir, file))
        df[parameter['name'] + "_" + parameter['level']] = values.flatten()

    print(f"Number of files: {len(files)}")
    sns.heatmap(df.corr(), annot=True)
    plt.savefig(os.path.join(Path(__file__).parent, "gfs_heatmap.png"))
    plt.close()


def prepare_synop_csv(localisation_name, code_fallback, features):
    download_list_of_station()
    localisation_code, name = get_localisation_id(localisation_name, code_fallback)
    localisation_code = localisation_code
    localisation_name = args.localisation_name
    if localisation_name is None:
        localisation_name = name
    process_all_data(2001, 2021, str(localisation_code), localisation_name, output_dir=SYNOP_DATASETS_DIRECTORY, columns=features)


def explore_synop_correlations(data, features, localisation_name):
    if os.path.exists(os.path.join(Path(__file__).parent, f"synop_{localisation_name}_heatmap.png")):
        return
    data = data[list(list(zip(*features))[1])]
    sns.heatmap(data.corr(), annot=True)
    plt.show()
    plt.savefig(os.path.join(Path(__file__).parent, f"synop_{localisation_name}_heatmap.png"))
    plt.close()


def explore_synop_patterns(data, relevant_features, localisation_name):
    features_with_nans = []
    for feature in relevant_features:
        plot_dir = os.path.join('plots-synop', localisation_name, feature[1])
        if not os.path.exists(plot_dir):
            values = data[feature[1]].to_numpy()
            if np.isnan(np.sum(values)):
                features_with_nans.append(feature[1])
            sns.boxplot(x=values).set_title(f"{feature[1]}")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, 'plot.png'))
            plt.close()
    if len(features_with_nans):
        logger = get_logger(os.path.join("explore_results", 'logs.log'))
        logger.info(f"Nans in features:\n {[f'{feature}, ' for feature in features_with_nans]}")


def explore_synop(localisation_name, code_fallback):
    relevant_features = [f for f in SYNOP_FEATURES if f[1] not in ['year', 'month', 'day', 'hour']]
    synop_file = f"{localisation_name}_{code_fallback}_data.csv"
    if not os.path.exists(os.path.join(SYNOP_DATASETS_DIRECTORY, synop_file)):
        prepare_synop_csv(localisation_name, code_fallback, SYNOP_FEATURES)

    data, _, _ = prepare_synop_dataset(synop_file, list(list(zip(*relevant_features))[1]), norm=False,
                                       dataset_dir=SYNOP_DATASETS_DIRECTORY)

    explore_synop_correlations(data, relevant_features, localisation_name)
    explore_synop_patterns(data, relevant_features, localisation_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--localisation_name', help='Localisation name for which to get data', default="WARSZAWA-OKECIE", type=str)
    parser.add_argument('--code_fallback', help='Localisation code as a fallback if name is not found', default=375,
                        type=int)

    args = parser.parse_args()

    explore_data_for_each_param()
    explore_gfs_correlations()
    explore_synop(args.localisation_name, args.code_fallback)

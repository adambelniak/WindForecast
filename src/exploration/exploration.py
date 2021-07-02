import os
import re

from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from gfs_archive_0_25.gfs_processor.own_logger import get_logger
from wind_forecast.util.utils import get_available_numpy_files
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
    for parameter in tqdm(GFS_PARAMETERS):
        param_dir = os.path.join(gfs_dataset_dir, parameter['name'], parameter['level'])
        if os.path.exists(param_dir):
            for offset in tqdm(range(3, 120, 3)):
                plot_dir = os.path.join('plots', parameter['name'], parameter['level'], str(offset))
                if not os.path.exists(os.path.join(plot_dir, 'plot.png')):
                    logger = get_logger(os.path.join("explore_results", parameter['name'], parameter['level'], str(offset), 'logs.log'))
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


def explore_corelations():
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
    plt.show()


if __name__ == "__main__":
    explore_data_for_each_param()
    # explore_corelations()

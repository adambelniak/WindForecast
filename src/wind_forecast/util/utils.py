import os
import re

import numpy as np
import pytz
from tqdm import tqdm


from gfs_archive_0_25.utils import prep_zeros_if_needed

GFS_DATASET_DIR = "D:\\WindForecast\\output_np2"


def convert_wind(single_gfs, u_wind_label, v_wind_label):
    single_gfs["velocity"] = np.sqrt(single_gfs[u_wind_label] ** 2 + single_gfs[v_wind_label] ** 2)
    single_gfs = single_gfs.drop([u_wind_label, v_wind_label], axis=1)

    return single_gfs


def utc_to_local(date):
    return date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Warsaw')).replace(tzinfo=None)


def get_available_numpy_files(features, offset, directory):
    result = None
    matcher = re.compile(rf".*f{prep_zeros_if_needed(str(offset), 2)}.*")
    for feature in tqdm(features):
        files = [f.name for f in os.scandir(os.path.join(directory, feature['name'], feature['level'])) if matcher.match(f.name)]
        result = np.intersect1d(result, np.array(files)) if result is not None else np.array(files)

    return result


def declination_of_earth(date):
    day_of_year = date.timetuple().tm_yday
    return 23.45*np.sin(np.deg2rad(360.0*(283.0+day_of_year)/365.0))

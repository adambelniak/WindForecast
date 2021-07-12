import math
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pytz
from scipy.interpolate import interpolate
from tqdm import tqdm


from gfs_archive_0_25.utils import prep_zeros_if_needed, get_nearest_coords
from gfs_common.common import GFS_SPACE
from wind_forecast.consts import NETCDF_FILE_REGEX
from wind_forecast.util.logging import log

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


def date_from_gfs_np_file(filename):
    date_matcher = re.match(NETCDF_FILE_REGEX, filename)

    date_from_filename = date_matcher.group(1)
    year = int(date_from_filename[:4])
    month = int(date_from_filename[5:7])
    day = int(date_from_filename[8:10])
    run = int(date_matcher.group(2))
    offset = int(date_matcher.group(3))
    forecast_date = utc_to_local(datetime(year, month, day) + timedelta(hours=run + offset))
    return forecast_date


def declination_of_earth(date):
    day_of_year = date.timetuple().tm_yday
    return 23.45*np.sin(np.deg2rad(360.0*(283.0+day_of_year)/365.0))


def initialize_mean_and_std(list_IDs, train_parameters, dim):
    log.info("Calculating std and mean for a dataset")
    means = []
    stds = []
    for param in tqdm(train_parameters):
        sum, sqr_sum = 0, 0
        for id in list_IDs:
            values = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], id))
            sum += np.sum(values)
            sqr_sum += pow(sum, 2)

        mean = sum / (len(list_IDs) * dim[0] * dim[1])
        means.append(mean)
        stds.append(math.sqrt(sqr_sum / (len(list_IDs) * dim[0] * dim[1]) - pow(mean, 2)))

    return means, stds

def initialize_mean_and_std_for_wind_target(list_IDs, dim):
    log.info("Calculating std and mean for a dataset")

    sum, sqr_sum = 0, 0
    for id in list_IDs:
        values_v = np.load(os.path.join(GFS_DATASET_DIR, 'V GRD', 'HTGL_10', id))
        values_u = np.load(os.path.join(GFS_DATASET_DIR, 'U GRD', 'HTGL_10', id))
        velocity = math.sqrt(values_v ** 2 + values_u ** 2)
        sum += np.sum(velocity)
        sqr_sum += pow(sum, 2)

    mean = sum / (len(list_IDs) * dim[0] * dim[1])

    return mean, math.sqrt(sqr_sum / (len(list_IDs) * dim[0] * dim[1]) - pow(mean, 2))


def get_point_from_GFS_slice_for_coords(gfs_data, latitude, longitude):
    coords = get_nearest_coords(latitude, longitude)
    lat, lon = coords[0][0], coords[1][0]
    lat_index = int((GFS_SPACE.nlat - lat) * 4)
    lon_index = int((GFS_SPACE.elon - lon) * 4)

    return interpolate.interp2d(coords[0], coords[1], gfs_data[lat_index:lat_index+2, lon_index:lon_index+2])(latitude, longitude).item()


def target_param_to_gfs_name_level(target_param):
    return {
        "temperature": [{
            "name": "TMP",
            "level": "HTGL_2"
        }],
        "wind_velocity": [{
            "name": "V GRD",
            "level": "HTGL_10"
          },
          {
            "name": "U GRD",
            "level": "HTGL_10"
          }
        ]
    }[target_param]
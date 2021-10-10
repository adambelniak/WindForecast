import math
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from wind_forecast.preprocess.synop.consts import SYNOP_FEATURES
from gfs_common.common import GFS_SPACE
from typing import Union
from scipy.interpolate import interpolate
from gfs_archive_0_25.gfs_processor.Coords import Coords
from gfs_archive_0_25.gfs_processor.consts import FINAL_NUMPY_FILENAME_FORMAT
from wind_forecast.consts import NETCDF_FILE_REGEX
from gfs_archive_0_25.utils import get_nearest_coords
from wind_forecast.util.common_util import prep_zeros_if_needed
from wind_forecast.util.logging import log

GFS_DATASET_DIR = os.environ.get('GFS_DATASET_DIR')


def convert_wind(single_gfs, u_wind_label, v_wind_label):
    single_gfs["velocity"] = np.sqrt(single_gfs[u_wind_label] ** 2 + single_gfs[v_wind_label] ** 2)
    single_gfs = single_gfs.drop([u_wind_label, v_wind_label], axis=1)

    return single_gfs


def get_available_numpy_files(features, offset, directory):
    result = None
    matcher = re.compile(rf".*f{prep_zeros_if_needed(str(offset), 2)}.*")
    for feature in tqdm(features):
        files = [f.name for f in os.scandir(os.path.join(directory, feature['name'], feature['level'])) if
                 matcher.match(f.name)]
        result = np.intersect1d(result, np.array(files)) if result is not None else np.array(files)

    return result.tolist()


def date_from_gfs_np_file(filename):
    date_matcher = re.match(NETCDF_FILE_REGEX, filename)

    date_from_filename = date_matcher.group(1)
    year = int(date_from_filename[:4])
    month = int(date_from_filename[5:7])
    day = int(date_from_filename[8:10])
    run = int(date_matcher.group(5))
    offset = int(date_matcher.group(6))
    forecast_date = datetime(year, month, day) + timedelta(hours=run + offset)
    return forecast_date


def get_GFS_values_for_sequence(file_id, param, sequence_length, subregion_coords=None):
    values = [get_subregion_from_GFS_slice_for_coords(
        np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], file_id)), subregion_coords)]
    date_matcher = re.match(NETCDF_FILE_REGEX, file_id)
    offset = int(date_matcher.group(6))
    for frame in range(1, sequence_length):
        new_id = file_id.replace(f"f{prep_zeros_if_needed(str(offset), 2)}",
                                 f"f{prep_zeros_if_needed(str(offset + 3), 2)}")
        val = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], new_id))
        if subregion_coords is not None:
            val = get_subregion_from_GFS_slice_for_coords(val, subregion_coords)
        values.append(val)
    return values


def initialize_mean_and_std(list_IDs, train_parameters, dim: (int, int), subregion_coords=None):
    log.info("Calculating std and mean for a dataset")
    means = []
    stds = []
    for param in tqdm(train_parameters):
        sum, sqr_sum = 0, 0
        for id in tqdm(list_IDs):
            values = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], id))
            if subregion_coords is not None:
                values = get_subregion_from_GFS_slice_for_coords(values, subregion_coords)
            sum += np.sum(values)
            sqr_sum += np.sum(np.power(values, 2))

        mean = sum / (len(list_IDs) * dim[0] * dim[1])
        means.append(mean)
        stds.append(math.sqrt(sqr_sum / (len(list_IDs) * dim[0] * dim[1]) - pow(mean, 2)))

    return means, stds


def initialize_mean_and_std_for_sequence(list_IDs, train_parameters, dim: (int, int), sequence_length: int,
                                         subregion_coords=None):
    log.info("Calculating std and mean for a dataset")
    means = []
    stds = []
    for param in tqdm(train_parameters):
        sum, sqr_sum = 0, 0
        for id in tqdm(list_IDs):
            values = np.squeeze(get_GFS_values_for_sequence(id, param, sequence_length, subregion_coords))
            sum += np.sum(values)
            sqr_sum += np.sum(np.power(values, 2))

        mean = sum / (len(list_IDs) * sequence_length * dim[0] * dim[1])
        means.append(mean)
        stds.append(math.sqrt(sqr_sum / (len(list_IDs) * sequence_length * dim[0] * dim[1]) - pow(mean, 2)))

    return means, stds


def initialize_min_max(list_IDs: [str], train_parameters, subregion_coords=None):
    log.info("Calculating min and max for a dataset")
    mins = []
    maxes = []
    for param in tqdm(train_parameters):
        min, max = sys.float_info.max, sys.float_info.min
        for id in list_IDs:
            values = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], id))
            if subregion_coords is not None:
                values = get_subregion_from_GFS_slice_for_coords(values, subregion_coords)
            min = min(np.min(values), min)
            max = max(np.max(values), max)

        mins.append(min)
        maxes.append(max)

    return mins, maxes


def initialize_min_max_for_sequence(list_IDs: [str], train_parameters, sequence_length: int, subregion_coords=None):
    log.info("Calculating min and max for the GFS dataset")
    mins = []
    maxes = []
    for param in tqdm(train_parameters):
        min, max = sys.float_info.max, sys.float_info.min
        for id in list_IDs:
            values = np.squeeze(get_GFS_values_for_sequence(id, param, sequence_length, subregion_coords))
            min = min(values, min)
            max = max(values, max)

        mins.append(min)
        maxes.append(max)

    return mins, maxes


def initialize_mean_and_std_for_wind_target(list_IDs, dim):
    log.info("Calculating std and mean for the GFS dataset")

    sum, sqr_sum = 0, 0
    for id in list_IDs:
        values_v = np.load(os.path.join(GFS_DATASET_DIR, 'V GRD', 'HTGL_10', id))
        values_u = np.load(os.path.join(GFS_DATASET_DIR, 'U GRD', 'HTGL_10', id))
        velocity = math.sqrt(values_v ** 2 + values_u ** 2)
        sum += np.sum(velocity)
        sqr_sum += pow(sum, 2)

    mean = sum / (len(list_IDs) * dim[0] * dim[1])

    return mean, math.sqrt(sqr_sum / (len(list_IDs) * dim[0] * dim[1]) - pow(mean, 2))


def initialize_GFS_list_IDs_for_sequence(list_IDs: [str], labels: pd.DataFrame, one_of_train_parameters,
                                         target_param: str, sequence_length: int):
    # filter out files, which are not continued by sufficient number of consecutive forecasts
    new_list = []
    list_IDs = sorted(list_IDs)
    param = one_of_train_parameters
    for id in list_IDs:
        date_matcher = re.match(NETCDF_FILE_REGEX, id)
        offset = int(date_matcher.group(6))
        exists = True
        for frame in range(1, sequence_length):
            new_id = id.replace(f"f{prep_zeros_if_needed(str(offset), 2)}",
                                f"f{prep_zeros_if_needed(str(offset + 3), 2)}")
            # check if gfs file exists
            if not os.path.exists(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], new_id)):
                exists = False
                break
            # check if synop label exists
            if len(labels[labels["date"] == date_from_gfs_np_file(id) + timedelta(hours=offset + 3)][
                       target_param]) == 0:
                exists = False
                break

            offset = offset + 3
        if exists:
            new_list.append(id)

    return new_list


def get_nearest_lat_lon_from_coords(gfs_coords: [[float]], original_coords: Coords):
    lat = gfs_coords[0][0] if abs(original_coords.nlat - gfs_coords[0][0]) <= abs(
        original_coords.nlat - gfs_coords[0][1]) else gfs_coords[0][1]
    lon = gfs_coords[1][0] if abs(original_coords.elon - gfs_coords[1][0]) <= abs(
        original_coords.elon - gfs_coords[1][1]) else gfs_coords[1][1]

    return lat, lon


def get_point_from_GFS_slice_for_coords(gfs_data: np.ndarray, coords: Coords):
    nearest_coords = get_nearest_coords(coords)
    lat, lon = get_nearest_lat_lon_from_coords(nearest_coords, coords)
    lat_index = int((GFS_SPACE.nlat - lat) * 4)
    lon_index = int((GFS_SPACE.elon - lon) * 4)

    return interpolate.interp2d(nearest_coords[0], nearest_coords[1],
                                gfs_data[lat_index:lat_index + 2, lon_index:lon_index + 2])(
        coords.nlat, coords.elon).item()


def get_subregion_from_GFS_slice_for_coords(gfs_data: np.ndarray, coords: Coords) -> np.ndarray:
    nearest_coords_NW = get_nearest_coords(Coords(coords.nlat, coords.nlat, coords.wlon, coords.wlon))
    nearest_coords_SE = get_nearest_coords(Coords(coords.slat, coords.slat, coords.elon, coords.elon))
    lat_NW, lon_NW = get_nearest_lat_lon_from_coords(nearest_coords_NW,
                                                     Coords(coords.nlat, coords.nlat, coords.wlon, coords.wlon))
    lat_SE, lon_SE = get_nearest_lat_lon_from_coords(nearest_coords_SE,
                                                     Coords(coords.slat, coords.slat, coords.elon, coords.elon))

    lat_index_start = int((GFS_SPACE.nlat - lat_NW) * 4)
    lat_index_end = int((GFS_SPACE.nlat - lat_SE) * 4)
    lon_index_start = int((GFS_SPACE.elon - lon_SE) * 4)
    lon_index_end = int((GFS_SPACE.elon - lon_NW) * 4)

    return gfs_data[lat_index_start:lat_index_end + 1, lon_index_start:lon_index_end + 1]


def get_dim_of_GFS_slice_for_coords(coords: Coords) -> (int, int):
    nearest_coords_NW = get_nearest_coords(Coords(coords.nlat, coords.nlat, coords.wlon, coords.wlon))
    nearest_coords_SE = get_nearest_coords(Coords(coords.slat, coords.slat, coords.elon, coords.elon))
    lat_NW, lon_NW = get_nearest_lat_lon_from_coords(nearest_coords_NW,
                                                     Coords(coords.nlat, coords.nlat, coords.wlon, coords.wlon))
    lat_SE, lon_SE = get_nearest_lat_lon_from_coords(nearest_coords_SE,
                                                     Coords(coords.slat, coords.slat, coords.elon, coords.elon))

    lat_index_start = int((GFS_SPACE.nlat - lat_NW) * 4)
    lat_index_end = int((GFS_SPACE.nlat - lat_SE) * 4)
    lon_index_start = int((GFS_SPACE.elon - lon_SE) * 4)
    lon_index_end = int((GFS_SPACE.elon - lon_NW) * 4)

    return lat_index_end - lat_index_start + 1, lon_index_end - lon_index_start + 1


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


def add_param_to_train_params(train_params: list, param: str):
    params = train_params
    if param not in list(list(zip(*train_params))[1]):
        if param not in list(list(zip(*SYNOP_FEATURES))[1]):
            raise ValueError(f"Param {param} is not known as a synop parameter.")
        for train_param in train_params:
            if train_param[1] == param:
                params.append(train_param)
                break

    return params


def get_GFS_filename(date, prediction_offset, exact_date_match):
    # value = [date, target_param]
    last_date_in_sequence = date - timedelta(
        hours=prediction_offset + (7 if date.month < 10 or date.month > 4 else 6))  # 00 run is available at 6 UTC
    day = last_date_in_sequence.day
    month = last_date_in_sequence.month
    year = last_date_in_sequence.year
    hour = int(last_date_in_sequence.hour)
    run = ['00', '06', '12', '18'][(hour // 6)]

    if exact_date_match:
        run_hour = int(run)
        prediction_hour = date.hour if date.hour > run_hour else date.hour + 24
        pred_offset = str(prediction_hour - run_hour)
    else:
        pred_offset = str((prediction_offset + 6) // 3 * 3)

    return FINAL_NUMPY_FILENAME_FORMAT.format(year, prep_zeros_if_needed(str(month), 1),
                                              prep_zeros_if_needed(str(day), 1), run,
                                              prep_zeros_if_needed(pred_offset, 2))


def match_gfs_with_synop_sequence(features: Union[list, np.ndarray], targets: list, lat: float, lon: float,
                                  prediction_offset: int, gfs_params: list, exact_date_match=False, return_GFS=True):
    gfs_values = []
    new_targets = []
    new_features = []

    for index, value in tqdm(enumerate(targets)):
        date = value[0]
        gfs_filename = get_GFS_filename(date, prediction_offset, exact_date_match)

        # check if there are forecasts available
        if all(os.path.exists(os.path.join(GFS_DATASET_DIR, param["name"], param["level"], gfs_filename)) for param in
               gfs_params):
            if return_GFS:
                val = []

                for param in gfs_params:
                    val.append(get_point_from_GFS_slice_for_coords(
                        np.load(
                            os.path.join(GFS_DATASET_DIR, param['name'], param['level'], gfs_filename)),
                        Coords(lat, lat, lon, lon)))

                gfs_values.append(val)
            new_targets.append(value[1])
            new_features.append(features[index])

    if return_GFS:
        return np.array(new_features), np.array(gfs_values), np.array(new_targets)
    return np.array(new_features), np.array(new_targets)


def match_gfs_with_synop_sequence2sequence(features: Union[list, np.ndarray], targets: list, lat: float, lon: float,
                                           prediction_offset: int, gfs_params: list, exact_date_match=False,
                                           return_GFS=True):
    gfs_values = []
    new_targets = []
    new_features = []
    gfs_values_cache = {}
    print("Matching GFS with synop data")
    for index, value in tqdm(enumerate(targets)):
        dates = value.loc[:, 'date']
        next_gfs_values = []
        exists = True
        for date in dates:
            cache_key = datetime.strftime(date, "%Y%m%d%H%H%M%M")
            gfs_filename = get_GFS_filename(date, prediction_offset, exact_date_match)

            # check if there are forecasts available
            if all(os.path.exists(os.path.join(GFS_DATASET_DIR, param["name"], param["level"], gfs_filename)) for param
                   in gfs_params):
                if return_GFS:
                    if gfs_values_cache.get(cache_key) is not None:
                        next_gfs_values.append(gfs_values_cache.get(cache_key))
                    else:
                        val = []

                        for param in gfs_params:
                            val.append(get_point_from_GFS_slice_for_coords(
                                np.load(
                                    os.path.join(GFS_DATASET_DIR, param['name'], param['level'], gfs_filename)),
                                Coords(lat, lat, lon, lon)))

                        next_gfs_values.append(val)
                        gfs_values_cache[cache_key] = val

            else:
                exists = False
                break
        if exists:  # all gfs forecasts are available
            gfs_values.append(next_gfs_values)
            new_features.append(features[index])
            new_targets.append(value.loc[:, value.columns != 'date'])
            gfs_values_cache[datetime.strftime(dates.iloc[0], "%Y%m%d%H%H%M%M")] = None

    if return_GFS:
        return np.array(new_features), np.array(gfs_values), np.array(new_targets)
    return np.array(new_features), np.array(new_targets)

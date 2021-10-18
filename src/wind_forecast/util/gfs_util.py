import math
import os
import re
import sys
from typing import Union

if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from wind_forecast.loaders.GFSLoader import GFSLoader
from wind_forecast.preprocess.synop.consts import SYNOP_FEATURES
from gfs_common.common import GFS_SPACE
from scipy.interpolate import interpolate
from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.consts import NETCDF_FILE_REGEX, DATE_KEY_REGEX
from gfs_archive_0_25.utils import get_nearest_coords
from wind_forecast.util.common_util import prep_zeros_if_needed, local_to_utc, NormalizationType
from wind_forecast.util.logging import log

GFS_DATASET_DIR = os.environ.get('GFS_DATASET_DIR')
GFS_DATASET_DIR = 'gfs_data' if GFS_DATASET_DIR is None else GFS_DATASET_DIR


def convert_wind(single_gfs, u_wind_label, v_wind_label):
    single_gfs["velocity"] = np.sqrt(single_gfs[u_wind_label] ** 2 + single_gfs[v_wind_label] ** 2)
    single_gfs = single_gfs.drop([u_wind_label, v_wind_label], axis=1)

    return single_gfs


def get_available_numpy_files(features, offset):
    result = None
    matcher = re.compile(rf".*f{prep_zeros_if_needed(str(offset), 2)}.*")
    for feature in tqdm(features):
        files = [f.name for f in os.scandir(os.path.join(GFS_DATASET_DIR, feature['name'], feature['level'])) if
                 matcher.match(f.name)]
        result = np.intersect1d(result, np.array(files)) if result is not None else np.array(files)

    return result.tolist()


def get_available_gfs_date_keys(features, prediction_offset: int, sequence_length: int, from_year: int = 2015):
    result = {}
    for feature in tqdm(features):
        for offset in range(prediction_offset, prediction_offset + sequence_length * 3, 3):
            meta_files_matcher = re.compile(rf"{feature['name']}_{feature['level']}_{prep_zeros_if_needed(str(offset), 2)}_meta\.pkl")
            pickle_dir = os.path.join(GFS_DATASET_DIR, 'pkl')
            print(f"Scanning {[pickle_dir]} looking for CMAX meta files.")
            meta_files = [f.name for f in tqdm(os.scandir(pickle_dir)) if meta_files_matcher.match(f.name)]
            date_keys = []
            for meta_file in meta_files:
                with open(os.path.join(pickle_dir, meta_file), 'rb') as f:
                    date_keys.extend(pickle.load(f))

            date_keys.sort()
            date_keys = filter(lambda item: int(item[:4]) >= from_year, list(set(date_keys)))
            result[str(offset)] = np.intersect1d(result[str(offset)], np.array(date_keys)) if str(offset) in result \
                else np.array(date_keys)

    return result


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


def date_from_gfs_date_key(date_key):
    date_matcher = re.match(DATE_KEY_REGEX, date_key)

    year = int(date_matcher.group(1))
    month = int(date_matcher.group(2))
    day = int(date_matcher.group(3))
    hour = int(date_matcher.group(4))
    date = datetime(year, month, day, hour)
    return date


def get_GFS_values_for_sequence(date_key, param, sequence_length: int, prediction_offset: int, subregion_coords: Coords = None):
    gfs_loader = GFSLoader()
    values = [get_subregion_from_GFS_slice_for_coords(gfs_loader.get_gfs_image(date_key, param, prediction_offset), subregion_coords)]
    for frame in range(1, sequence_length):
        new_date_key = GFSLoader.get_date_key(date_from_gfs_date_key(date_key) + timedelta(hours=3 * frame))
        val = gfs_loader.get_gfs_image(new_date_key, param, prediction_offset + 3 * frame)
        if subregion_coords is not None:
            val = get_subregion_from_GFS_slice_for_coords(val, subregion_coords)
        values.append(val)

    return values


def initialize_mean_and_std(date_keys, train_parameters, dim: (int, int), prediction_offset: int, subregion_coords=None):
    log.info("Calculating std and mean for a dataset")
    means = []
    stds = []
    gfs_loader = GFSLoader()
    for param in tqdm(train_parameters):
        sum, sqr_sum = 0, 0
        for date_key in tqdm(date_keys):
            values = gfs_loader.get_gfs_image(date_key, param, prediction_offset)
            if subregion_coords is not None:
                values = get_subregion_from_GFS_slice_for_coords(values, subregion_coords)
            sum += np.sum(values)
            sqr_sum += np.sum(np.power(values, 2))

        mean = sum / (len(date_keys) * dim[0] * dim[1])
        means.append(mean)
        stds.append(math.sqrt(sqr_sum / (len(date_keys) * dim[0] * dim[1]) - pow(mean, 2)))

    return means, stds


def initialize_mean_and_std_for_sequence(date_keys: dict, train_parameters, dim: (int, int), sequence_length: int, prediction_offset: int,
                                         subregion_coords: Coords = None):
    log.info("Calculating std and mean for a dataset")
    means = []
    stds = []
    for param in tqdm(train_parameters):
        sum, sqr_sum = 0, 0
        for id in tqdm(date_keys[prediction_offset]):
            values = np.squeeze(get_GFS_values_for_sequence(id, param, sequence_length, prediction_offset, subregion_coords))
            sum += np.sum(values)
            sqr_sum += np.sum(np.power(values, 2))

        mean = sum / (len(date_keys) * sequence_length * dim[0] * dim[1])
        means.append(mean)
        stds.append(math.sqrt(sqr_sum / (len(date_keys) * sequence_length * dim[0] * dim[1]) - pow(mean, 2)))

    return means, stds


def initialize_min_max(date_keys: [str], train_parameters, prediction_offset: int, subregion_coords=None):
    log.info("Calculating min and max for a dataset")
    mins = []
    maxes = []
    gfs_loader = GFSLoader()
    for param in tqdm(train_parameters):
        min, max = sys.float_info.max, sys.float_info.min
        for date_key in date_keys:
            values = gfs_loader.get_gfs_image(date_key, param, prediction_offset)
            if subregion_coords is not None:
                values = get_subregion_from_GFS_slice_for_coords(values, subregion_coords)
            min = min(np.min(values), min)
            max = max(np.max(values), max)

        mins.append(min)
        maxes.append(max)

    return mins, maxes


def initialize_min_max_for_sequence(list_IDs: [str], train_parameters, sequence_length: int, prediction_offset: int, subregion_coords=None):
    log.info("Calculating min and max for the GFS dataset")
    mins = []
    maxes = []
    for param in tqdm(train_parameters):
        min, max = sys.float_info.max, sys.float_info.min
        for id in list_IDs:
            values = np.squeeze(get_GFS_values_for_sequence(id, param, sequence_length, prediction_offset, subregion_coords))
            min = min(values, min)
            max = max(values, max)

        mins.append(min)
        maxes.append(max)

    return mins, maxes


def initialize_GFS_date_keys_for_sequence(date_keys: [str], labels: pd.DataFrame, train_params: [dict], target_param: str, sequence_length: int):
    # filter out date_keys, which are not continued by sufficient number of consecutive forecasts
    new_list = []
    date_keys = sorted(date_keys)
    gfs_loader = GFSLoader()
    for date_key in date_keys:
        date_matcher = re.match(NETCDF_FILE_REGEX, date_key)
        offset = int(date_matcher.group(6)) + 3
        exists = True
        for frame in range(1, sequence_length):
            new_date_key = GFSLoader.get_date_key(date_from_gfs_date_key(date_key) + timedelta(hours=offset))
            # check if gfs file exists
            if not all(gfs_loader.get_gfs_image(new_date_key, param, offset) is not None for param in train_params):
                exists = False
                break
            # check if synop label exists
            if len(labels[labels["date"] == date_from_gfs_date_key(date_key)
                          + timedelta(hours=offset)][target_param]) == 0:
                exists = False
                break

            offset = offset + 3
        if exists:
            new_list.append(date_key)

    return new_list


def normalize_gfs_data(gfs_data: np.ndarray, normalization_type: NormalizationType):
    if normalization_type == NormalizationType.STANDARD:
        return (gfs_data - np.mean(gfs_data)) / np.std(gfs_data)
    else:
        return (gfs_data - np.min(gfs_data)) / (
                    np.max(gfs_data) - np.min(gfs_data))


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


def get_indices_of_GFS_slice_for_coords(coords: Coords):
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

    return lat_index_start, lat_index_end, lon_index_start, lon_index_end


def get_subregion_from_GFS_slice_for_coords(gfs_data: np.ndarray, coords: Coords) -> np.ndarray:
    lat_index_start, lat_index_end, lon_index_start, lon_index_end = get_indices_of_GFS_slice_for_coords(coords)

    return gfs_data[lat_index_start:lat_index_end + 1, lon_index_start:lon_index_end + 1]


def get_dim_of_GFS_slice_for_coords(coords: Coords) -> (int, int):
    lat_index_start, lat_index_end, lon_index_start, lon_index_end = get_indices_of_GFS_slice_for_coords(coords)

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


def get_forecast_date_and_offset_for_prediction_date(date, prediction_offset):
    utc_date = local_to_utc(date)
    forecast_start_date = utc_date - timedelta(hours=prediction_offset + 5)  # 00 run is available at 5 UTC
    hour = int(forecast_start_date.hour)
    forecast_start_date = forecast_start_date - timedelta(hours=hour % 6)

    pred_offset = (prediction_offset + 6) // 3 * 3
    return forecast_start_date + timedelta(hours=pred_offset), pred_offset


def match_gfs_with_synop_sequence(features: Union[list, np.ndarray], targets: list, lat: float, lon: float,
                                  prediction_offset: int, gfs_params: list, exact_date_match=False, return_GFS=True):
    gfs_values = []
    new_targets = []
    new_features = []
    gfs_loader = GFSLoader()
    removed_indices = []
    print("Matching GFS with synop data")

    for index, value in tqdm(enumerate(targets)):
        date = value[0]
        gfs_date, gfs_offset = get_forecast_date_and_offset_for_prediction_date(date, prediction_offset,
                                                                                exact_date_match)
        gfs_date_key = gfs_loader.get_date_key(gfs_date)

        # check if there are forecasts available
        if all(gfs_loader.get_gfs_image(gfs_date_key, param, gfs_offset) is not None for param in gfs_params):
            if return_GFS:
                val = []

                for param in gfs_params:
                    val.append(
                        get_point_from_GFS_slice_for_coords(gfs_loader.get_gfs_image(gfs_date_key, param, gfs_offset),
                                                            Coords(lat, lat, lon, lon)))

                gfs_values.append(val)
            new_targets.append(value[1])
            new_features.append(features[index])
        else:
            removed_indices.append(index)

    if return_GFS:
        return np.array(new_features), np.array(gfs_values), np.array(new_targets), removed_indices
    return np.array(new_features), np.array(new_targets), removed_indices


def match_gfs_with_synop_sequence2sequence(synop_features: list, targets: list, lat: float, lon: float,
                                           prediction_offset: int, gfs_params: list,
                                           return_GFS=True):
    gfs_values = []
    new_targets = []
    new_synop_features = []
    removed_indices = []
    gfs_loader = GFSLoader()
    print("Matching GFS with synop data")
    for index, value in tqdm(enumerate(targets)):
        dates = value.loc[:, 'date']
        next_gfs_values = []
        exists = True
        for date in dates:
            gfs_date, gfs_offset = get_forecast_date_and_offset_for_prediction_date(date, prediction_offset)
            gfs_date_key = gfs_loader.get_date_key(gfs_date)
            # check if there are forecasts available
            if all(gfs_loader.get_gfs_image(gfs_date_key, param, gfs_offset) is not None for param in gfs_params):
                if return_GFS:
                    val = []
                    for param in gfs_params:
                        val.append(get_point_from_GFS_slice_for_coords(
                            gfs_loader.get_gfs_image(gfs_date_key, param, gfs_offset),
                            Coords(lat, lat, lon, lon)))

                    next_gfs_values.append(val)

            else:
                removed_indices.append(index)
                exists = False
                break
        if exists:  # all gfs forecasts are available
            gfs_values.append(next_gfs_values)
            new_synop_features.append(synop_features[index])
            new_targets.append(value)

    if return_GFS:
        return new_synop_features, np.array(gfs_values), new_targets, removed_indices
    return new_synop_features, new_targets, removed_indices

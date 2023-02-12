import math
import os
import re
import sys
from typing import Union, Dict

from statsmodels.tsa.seasonal import STL

from wind_forecast.loaders.GFSInterpolatedLoader import GFSInterpolatedLoader, Interpolator
from wind_forecast.util.common_util import NormalizationType

if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from wind_forecast.loaders.GFSLoader import GFSLoader
from synop.consts import SYNOP_FEATURES
from gfs_common.common import GFS_SPACE
from scipy.interpolate import CubicSpline
from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.consts import NETCDF_FILE_REGEX, DATE_KEY_REGEX
from gfs_archive_0_25.utils import get_nearest_coords
from util.util import prep_zeros_if_needed

from wind_forecast.util.logging import log

GFS_DATASET_DIR = os.environ.get('GFS_DATASET_DIR')
GFS_DATASET_DIR = 'gfs_data' if GFS_DATASET_DIR is None else GFS_DATASET_DIR


class GFSUtil:
    def __init__(self, target_coords: Coords, past_sequence_length: int, future_sequence_length: int,
                 prediction_offset: int, train_params: list) -> None:
        super().__init__()
        self.target_coords = target_coords
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.prediction_offset = prediction_offset
        self.train_params = train_params
        self.features_names = [f"{f['name']}_{f['level']}" for f in self.train_params]
        self.interpolator = Interpolator(self.target_coords)
        self.gfs_interpolated_loader = GFSInterpolatedLoader(self.target_coords)

    def match_gfs_with_synop_sequence(self, synop_data: pd.DataFrame, synop_data_indices: list) -> (list, pd.DataFrame):
        new_synop_indices = []
        gfs_data = pd.DataFrame(columns=[*self.features_names, 'date'])
        synop_dates_series = synop_data['date']

        log.info("Matching GFS with synop data")

        for index in tqdm(synop_data_indices):
            sequence_dates = synop_dates_series.loc[index:index + self.past_sequence_length - 1]

            gfs_values_for_sequence = []
            all_values_available = True
            for sequence_index, date in enumerate(sequence_dates):
                if len(gfs_data[gfs_data['date'] == date]) > 0:
                    continue

                if len(gfs_values_for_sequence) == 0:
                    # match GFS params for past sequences
                    gfs_values_for_past_sequence = self.get_next_gfs_values(sequence_dates, self.train_params, False)
                    if gfs_values_for_past_sequence is None:
                        all_values_available = False
                        break
                    gfs_values_for_sequence = gfs_values_for_past_sequence

                gfs_values = gfs_values_for_sequence[sequence_index]
                gfs_values = np.append(gfs_values, date)
                absolute_index = synop_data[synop_data['date'] == date].index[0]
                gfs_data.loc[absolute_index] = gfs_values

            if all_values_available:
                new_synop_indices.append(index)
        gfs_data[self.features_names] = gfs_data[self.features_names].astype(float)
        return new_synop_indices, gfs_data

    def match_gfs_with_synop_sequence2sequence(self, synop_data: pd.DataFrame,
                                               synop_data_indices: list) -> (list, pd.DataFrame):
        new_synop_indices = []
        gfs_data = pd.DataFrame(columns=[*self.features_names, 'date'])
        synop_dates_series = synop_data['date']

        log.info("Matching GFS with synop data")

        for index in tqdm(synop_data_indices):
            past_dates = synop_dates_series.loc[index:index + self.past_sequence_length - 1]
            future_dates = synop_dates_series.loc[
                               index + self.past_sequence_length + self.prediction_offset
                               :index + self.past_sequence_length + self.prediction_offset + self.future_sequence_length - 1]
            sequence_dates = pd.concat([past_dates, future_dates])

            gfs_values_for_sequence = []
            all_values_available = True
            for sequence_index, date in enumerate(sequence_dates):
                if len(gfs_data[gfs_data['date'] == date]) > 0:
                    continue

                if len(gfs_values_for_sequence) == 0:
                    # match GFS params for past sequences
                    gfs_values_for_past_sequence = self.get_next_gfs_values(past_dates, self.train_params, False)
                    if gfs_values_for_past_sequence is None:
                        all_values_available = False
                        break

                    # match GFS params for future sequences
                    gfs_values_for_future_sequence = self.get_next_gfs_values(future_dates, self.train_params, True)
                    if gfs_values_for_future_sequence is None:
                        all_values_available = False
                        break

                    gfs_values_for_sequence = np.concatenate([gfs_values_for_past_sequence, gfs_values_for_future_sequence], -2)

                gfs_values = gfs_values_for_sequence[sequence_index]
                gfs_values = np.append(gfs_values, date)
                absolute_index = synop_data[synop_data['date'] == date].index[0]
                gfs_data.loc[absolute_index] = gfs_values

            if all_values_available:
                new_synop_indices.append(index)
        gfs_data[self.features_names] = gfs_data[self.features_names].astype(float)
        return new_synop_indices, gfs_data

    def get_next_gfs_values(self, dates: [datetime], gfs_params: list, future_dates: bool):
        next_gfs_values = []
        first_date = pd.Timestamp(dates.values[0]).to_pydatetime()
        last_date = pd.Timestamp(dates.values[-1]).to_pydatetime()
        gfs_values = []
        x_values = []
        # there should be at least and at most 1 gfs forecast ahead and behind which will help in interpolation
        dates_3_hours_before = [first_date - timedelta(hours=3),
                                first_date - timedelta(hours=2),
                                first_date - timedelta(hours=1)]
        dates_3_hours_after = [last_date + timedelta(hours=1),
                               last_date + timedelta(hours=2),
                               last_date + timedelta(hours=3)]
        for index, date in enumerate(dates_3_hours_before):
            value = self.get_gfs_values_for_date(date, future_dates, gfs_params, first_date)
            if value is not None:
                if len(value) != 0:
                    x_values.append(index - 3)
                    gfs_values.append(value)
                    break
                elif index == 2:
                    return None

        for index, date in enumerate(dates):
            # get real forecast values for hours which match the GFS forecast hour (if mod_offset == 0)
            value = self.get_gfs_values_for_date(date, future_dates, gfs_params, first_date)
            if value is not None:
                # Date does match
                if len(value) != 0:
                    gfs_values.append(value)
                    x_values.append(index)
                else:
                    return None

        for index, date in enumerate(dates_3_hours_after):
            value = self.get_gfs_values_for_date(date, future_dates, gfs_params, first_date)
            if value is not None:
                if len(value) != 0:
                    x_values.append(index + len(dates))
                    gfs_values.append(value)
                    break
            elif index == 2:
                return None

        if len(x_values) < 2:
            return None

        for index, param in enumerate(gfs_params):
            if param['interpolation'] == 'polynomial':
                interpolator = CubicSpline(x_values, [gfs_values[i][index] for i in range(len(gfs_values))])
                values = interpolator(np.arange(len(dates)))
            else:
                values = np.interp(np.arange(len(dates)), x_values, [gfs_values[i][index] for i in range(len(gfs_values))])

            if 'min' in param:
                values = [param['min'] if val < param['min'] else val for val in values]
            if 'max' in param:
                values = [param['max'] if val > param['max'] else val for val in values]

            next_gfs_values.append(values)

        return np.array(next_gfs_values).transpose([1, 0])

    """
        Return values for all params only if there is a forecast for this exact date and hour.
        Otherwise, if the date does not match the forecast date return None, but if the date match and still there's no 
        forecast, return an empty array.
        current_date is when we start forecasting - we can't use the forecast from the future.
    """
    def get_gfs_values_for_date(self, date: datetime, future_date: bool, gfs_params: list, current_date: datetime):
        if future_date:
            offset = self.prediction_offset + int(divmod((date - current_date).total_seconds(), 3600)[0])
            offset = max(3, offset)
        else:
            # for past dates we assume we can use any closest forecast
            offset = 3
        gfs_date, gfs_offset, mod_offset = get_forecast_date_and_offset_for_prediction_date(date, offset)

        vals = []
        if mod_offset == 0:
            for param in gfs_params:
                value = self.gfs_interpolated_loader.get_gfs_value_for_date(gfs_date, param, gfs_offset)

                if value is None:
                    return []

                vals.append(value)

            return vals

        else:
            return None


def get_available_numpy_files(features: list, offset: int):
    result = None
    matcher = re.compile(rf".*f{prep_zeros_if_needed(str(offset), 2)}.*")
    for feature in tqdm(features):
        files = [f.name for f in os.scandir(os.path.join(GFS_DATASET_DIR, feature['name'], feature['level'])) if
                 matcher.match(f.name)]
        result = np.intersect1d(result, np.array(files)) if result is not None else np.array(files)

    return result.tolist()


def get_available_gfs_date_keys(features: list, prediction_offset: int, sequence_length: int,
                                from_year: int = 2015) -> Dict:
    result = {}
    for feature in tqdm(features):
        for offset in range(prediction_offset, prediction_offset + sequence_length * 3, 3):
            meta_files_matcher = re.compile(
                rf"{feature['name']}_{feature['level']}_{prep_zeros_if_needed(str(offset), 2)}_meta\.pkl")
            pickle_dir = os.path.join(GFS_DATASET_DIR, 'pkl')
            log.info(f"Scanning {[pickle_dir]} looking for CMAX meta files.")
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


def date_from_gfs_np_file(filename: str):
    date_matcher = re.match(NETCDF_FILE_REGEX, filename)

    date_from_filename = date_matcher.group(1)
    year = int(date_from_filename[:4])
    month = int(date_from_filename[5:7])
    day = int(date_from_filename[8:10])
    run = int(date_matcher.group(5))
    offset = int(date_matcher.group(6))
    forecast_date = datetime(year, month, day) + timedelta(hours=run + offset)
    return forecast_date


def date_from_gfs_date_key(date_key: str):
    date_matcher = re.match(DATE_KEY_REGEX, date_key)

    year = int(date_matcher.group(1))
    month = int(date_matcher.group(2))
    day = int(date_matcher.group(3))
    hour = int(date_matcher.group(4))
    date = datetime(year, month, day, hour)
    return date


def get_GFS_values_for_sequence(date_key: str, param: Dict, sequence_length: int, prediction_offset: int,
                                subregion_coords: Coords = None):
    gfs_loader = GFSLoader()
    values = [get_subregion_from_GFS_slice_for_coords(gfs_loader.get_gfs_image(date_key, param, prediction_offset),
                                                      subregion_coords)]
    for frame in range(1, sequence_length):
        new_date_key = GFSLoader.get_date_key(date_from_gfs_date_key(date_key) + timedelta(hours=3 * frame))
        val = gfs_loader.get_gfs_image(new_date_key, param, prediction_offset + 3 * frame)
        if subregion_coords is not None:
            val = get_subregion_from_GFS_slice_for_coords(val, subregion_coords)
        values.append(val)

    return values


def initialize_mean_and_std(date_keys: Dict, train_parameters, dim: (int, int), prediction_offset: int,
                            subregion_coords: Coords = None):
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


def initialize_mean_and_std_for_sequence(date_keys: dict, train_parameters, dim: (int, int), sequence_length: int,
                                         prediction_offset: int,
                                         subregion_coords: Coords = None):
    log.info("Calculating std and mean for a dataset")
    means = []
    stds = []
    for param in tqdm(train_parameters):
        sum, sqr_sum = 0, 0
        for id in tqdm(date_keys[prediction_offset]):
            values = np.squeeze(
                get_GFS_values_for_sequence(id, param, sequence_length, prediction_offset, subregion_coords))
            sum += np.sum(values)
            sqr_sum += np.sum(np.power(values, 2))

        mean = sum / (len(date_keys) * sequence_length * dim[0] * dim[1])
        means.append(mean)
        stds.append(math.sqrt(sqr_sum / (len(date_keys) * sequence_length * dim[0] * dim[1]) - pow(mean, 2)))

    return means, stds


def initialize_min_max(date_keys: [str], train_parameters: list, prediction_offset: int,
                       subregion_coords: Coords = None):
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


def initialize_min_max_for_sequence(list_IDs: [str], train_parameters: list, sequence_length: int,
                                    prediction_offset: int,
                                    subregion_coords: Coords = None):
    log.info("Calculating min and max for the GFS dataset")
    mins = []
    maxes = []
    for param in tqdm(train_parameters):
        min, max = sys.float_info.max, sys.float_info.min
        for id in list_IDs:
            values = np.squeeze(
                get_GFS_values_for_sequence(id, param, sequence_length, prediction_offset, subregion_coords))
            min = min(values, min)
            max = max(values, max)

        mins.append(min)
        maxes.append(max)

    return mins, maxes


def initialize_GFS_date_keys_for_sequence(date_keys: [str], labels: pd.DataFrame, train_params: [dict],
                                          target_param: str, sequence_length: int):
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


def normalize_gfs_data(gfs_data: pd.DataFrame, normalization_type: NormalizationType):
    if normalization_type == NormalizationType.STANDARD:
        data_mean = gfs_data.mean(axis=0)
        data_std = gfs_data.std(axis=0)
        return (gfs_data - data_mean) / data_std

    else:
        data_min = gfs_data.min(axis=0)
        data_max = gfs_data.max(axis=0)
        return (gfs_data - data_min) / (data_max - data_min)


def extend_wind_components(gfs_data: np.ndarray):
    velocity = np.apply_along_axis(lambda velocity: math.sqrt(velocity[0] ** 2 + velocity[1] ** 2), -1,
                                                  gfs_data)
    sin = np.apply_along_axis(lambda velocity: -velocity[1] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)), -1, gfs_data)
    cos = np.apply_along_axis(lambda velocity: -velocity[0] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)), -1, gfs_data)

    return velocity, sin, cos


def get_gfs_lat_lon_from_coords(gfs_coords: [[float]], original_coords: Coords):
    lat = gfs_coords[0][0] if abs(original_coords.nlat - gfs_coords[0][0]) <= abs(
        original_coords.nlat - gfs_coords[0][1]) else gfs_coords[0][1]
    lon = gfs_coords[1][0] if abs(original_coords.elon - gfs_coords[1][0]) <= abs(
        original_coords.elon - gfs_coords[1][1]) else gfs_coords[1][1]

    return lat, lon


def get_indices_of_GFS_slice_for_coords(coords: Coords):
    nearest_coords_NW = get_nearest_coords(Coords(coords.nlat, coords.nlat, coords.wlon, coords.wlon))
    nearest_coords_SE = get_nearest_coords(Coords(coords.slat, coords.slat, coords.elon, coords.elon))
    lat_NW, lon_NW = get_gfs_lat_lon_from_coords(nearest_coords_NW,
                                                 Coords(coords.nlat, coords.nlat, coords.wlon, coords.wlon))
    lat_SE, lon_SE = get_gfs_lat_lon_from_coords(nearest_coords_SE,
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


def target_param_to_gfs_name_level(target_param: str):
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
        ],
        "wind_direction": [{
            "name": "V GRD",
            "level": "HTGL_10"
        },
            {
                "name": "U GRD",
                "level": "HTGL_10"
            }
        ],
        "wind_gust": [{
            "name": "GUST",
            "level": "SFC_0"
        }],
        "pressure": [{
            "name": "PRES",
            "level": "SFC_0"
        }]
    }[target_param]


def get_gfs_target_param(target_param: str):
    return {
            'temperature': 'TMP_HTGL_2',
            'wind_velocity': 'wind-velocity',
            'pressure': 'PRES_SFC_0',
            'lower_clouds': 'T CDC_LCY_0'
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


def get_forecast_date_and_offset_for_prediction_date(date, prediction_offset: int):
    forecast_start_date = date - timedelta(hours=prediction_offset + 5)  # 00 run is available at 5 UTC
    hour = int(forecast_start_date.hour)
    forecast_start_date = forecast_start_date - timedelta(hours=hour % 6)
    real_prediction_offset = prediction_offset + 5 + hour % 6
    pred_offset = real_prediction_offset // 3 * 3
    return forecast_start_date + timedelta(hours=pred_offset), pred_offset, real_prediction_offset % 3


def decompose_gfs_data(gfs_data, features):
    for feature in features:
        series = gfs_data[feature]
        stl = STL(series, seasonal=35, period=24, trend=81, low_pass=25)
        res = stl.fit(inner_iter=1, outer_iter=10)
        O, T, S, R = res.observed, res.trend, res.seasonal, res.resid
        gfs_data[f"{feature}_T"] = T
        gfs_data[f"{feature}_S"] = S
        gfs_data[f"{feature}_R"] = R
        gfs_data.drop(columns=[feature], inplace=True)
    return gfs_data


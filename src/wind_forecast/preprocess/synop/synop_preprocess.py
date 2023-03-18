import datetime
import os
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import pandas as pd

from synop.consts import SYNOP_PERIODIC_FEATURES, VISIBILITY, DIRECTION_COLUMN, VELOCITY_COLUMN, GUST_COLUMN, \
    TEMPERATURE, HUMIDITY, DEW_POINT, PRESSURE
from wind_forecast.util.common_util import NormalizationType


def get_normalization_values(data: pd.DataFrame,
                             normalization_type: NormalizationType = NormalizationType.STANDARD) -> (pd.DataFrame, float, float):
    if normalization_type == NormalizationType.STANDARD:
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        return (data - data_mean) / data_std, data_mean, data_std
    else:
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        return (data - data_min) / (data_max - data_min), data_min, data_max


def prepare_synop_dataset(synop_file_name: str, features: list, norm=True,
                          dataset_dir=os.path.join(Path(__file__).parent, 'synop_data'),
                          from_year=2001, to_year=2021,
                          normalization_type: NormalizationType = NormalizationType.STANDARD,
                          decompose_periodic=True) \
        -> Union[Tuple[pd.DataFrame, float, float], pd.DataFrame]:
    synop_file_path = os.path.join(dataset_dir, synop_file_name)
    if not os.path.exists(synop_file_path):
        raise Exception(f"Dataset not found. Looked for {synop_file_path}")

    data = pd.read_csv(synop_file_path, usecols=features + ['year', 'month', 'day', 'hour'])
    data = data.dropna()
    if synop_file_name == 'WARSZAWA-OKECIE_352200375_data.csv':
        incorrect_gusts = data[data[GUST_COLUMN[1]] >= 22]
        incorrect_gusts = incorrect_gusts[incorrect_gusts[VELOCITY_COLUMN[1]] < 2]
        for index in incorrect_gusts.index:
            data.loc[index][GUST_COLUMN[1]] = data.loc[index-1][GUST_COLUMN[1]]

    features_to_check_for_zero_series = [feature for feature in features if feature in [
        VISIBILITY[1],
        DIRECTION_COLUMN[1],
        VELOCITY_COLUMN[1],
        GUST_COLUMN[1],
        TEMPERATURE[1],
        HUMIDITY[1],
        DEW_POINT[1],
        PRESSURE[1],
    ]]
    data = drop_zeros_series(data, features_to_check_for_zero_series, 12)

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

    first_date = datetime.datetime(year=from_year, month=1, day=1)
    last_date = datetime.datetime(year=to_year, month=1, day=1)

    data = data[(data['date'] >= first_date) & (data['date'] < last_date)]

    if decompose_periodic:
        data = decompose_periodic_features(data, features)
    if norm:
        data[features], mean_or_min, std_or_max = get_normalization_values(data[features].values, normalization_type)
        return data, mean_or_min, std_or_max

    return data


def decompose_periodic_features(data: pd.DataFrame, all_features: List[str]):
    for feature in SYNOP_PERIODIC_FEATURES:
        min = feature['min']
        max = feature['max']
        column = feature['column'][1]
        series_to_reduce = pd.to_numeric(data[column])
        period_argument = ((series_to_reduce - min) / (max - min)).astype(np.float64) * 2 * np.pi
        data.insert(data.columns.get_loc(column), f'{column}-cos', np.cos(period_argument).tolist())
        data.insert(data.columns.get_loc(column), f'{column}-sin', np.sin(period_argument).tolist())
        data.drop(columns=[column], inplace=True)
        all_features = modify_feature_names_after_periodic_reduction(all_features)
    return data


def drop_zeros_series(data: pd.DataFrame, features: [str], min_series_len_to_remove: int):
    indices_to_remove = []
    for feature in features:
        series_list = data[feature].values
        consecutive_zeros = 0
        for index, val in enumerate(series_list):
            if val == 0 and index != len(series_list) - 1:
                consecutive_zeros += 1
            else:
                if consecutive_zeros >= min_series_len_to_remove or val == 0:
                    indices_to_remove.extend(list(range(index - consecutive_zeros, index)))
                    if val == 0:
                        indices_to_remove.append(index)
                consecutive_zeros = 0

    return data.drop(data.index[indices_to_remove])


def modify_feature_names_after_periodic_reduction(features: list):
    new_features = features
    for feature in SYNOP_PERIODIC_FEATURES:
        index = new_features.index(feature['column'][1])
        new_features.insert(index, f'{feature["column"][1]}-cos')
        new_features.insert(index, f'{feature["column"][1]}-sin')
        new_features.remove(feature['column'][1])
    return new_features

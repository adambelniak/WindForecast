import datetime
import os
from typing import Dict, Union, Tuple
from statsmodels.tsa.seasonal import STL

import numpy as np
from pathlib import Path
import pandas as pd

from synop.consts import SYNOP_PERIODIC_FEATURES
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
                          normalization_type: NormalizationType = NormalizationType.STANDARD) \
        -> Union[Tuple[pd.DataFrame, float, float], pd.DataFrame]:
    synop_file_path = os.path.join(dataset_dir, synop_file_name)
    if not os.path.exists(synop_file_path):
        raise Exception(f"Dataset not found. Looked for {synop_file_path}")

    data = pd.read_csv(synop_file_path, usecols=features + ['year', 'month', 'day', 'hour'])
    data = data.dropna()

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

    first_date = datetime.datetime(year=from_year, month=1, day=1)
    last_date = datetime.datetime(year=to_year, month=1, day=1)

    data = data[(data['date'] >= first_date) & (data['date'] < last_date)]

    for feature in SYNOP_PERIODIC_FEATURES:
        min = feature['min']
        max = feature['max']
        column = feature['column'][1]
        series_to_reduce = data[column]
        period_argument = ((series_to_reduce - min) / (max - min)).astype(np.float64) * 2 * np.pi
        data[f'{column}-sin'] = np.sin(period_argument).tolist()
        data[f'{column}-cos'] = np.cos(period_argument).tolist()
        data.drop(columns=[column], inplace=True)
        features = get_feature_names_after_periodic_reduction(features)
    if norm:
        data[features], mean_or_min, std_or_max = get_normalization_values(data[features].values, normalization_type)
        return data, mean_or_min, std_or_max

    return data


def get_feature_names_after_periodic_reduction(features: list):
    for feature in SYNOP_PERIODIC_FEATURES:
        features.remove(feature['column'][1])
        features.append(f'{feature["column"][1]}-sin')
        features.append(f'{feature["column"][1]}-cos')
    return features


def decompose_synop_data(synop_data: pd.DataFrame, features: list):
    for feature in features:
        series = synop_data[feature]
        stl = STL(series, seasonal=35, period=24, trend=81, low_pass=25)
        res = stl.fit(inner_iter=1, outer_iter=10)
        O, T, S, R = res.observed, res.trend, res.seasonal, res.resid
        synop_data[f"{feature}_T"] = T
        synop_data[f"{feature}_S"] = S
        synop_data[f"{feature}_R"] = R
        synop_data.drop(columns=[feature], inplace=True)
    return synop_data


def resolve_synop_data(all_synop_data: pd.DataFrame, synop_data_indices: [int], length_of_sequence) -> pd.DataFrame:
    # Bear in mind that synop_data_indices are indices of FIRST synop in the sequence. Not all synop data exist in synop_data_indices because of that fact.
    all_indices = set(
        [item for sublist in [[index + frame for frame in range(0, length_of_sequence)] for index in synop_data_indices]
         for item in sublist])
    return all_synop_data.take(list(all_indices))


def normalize_synop_data_for_training(all_synop_data: pd.DataFrame, synop_data_indices: [int], features: [str],
                                      length_of_sequence: int, normalization_type: NormalizationType
                                      = NormalizationType.STANDARD) -> (pd.DataFrame, [str], Dict, Dict):
    all_relevant_synop_data = resolve_synop_data(all_synop_data, synop_data_indices, length_of_sequence)
    final_data = pd.DataFrame()
    mean_or_min_to_return = {}
    std_or_max_to_return = {}
    for feature in features:
        series_to_normalize = all_relevant_synop_data[feature]

        values, mean_or_min, std_or_max = get_normalization_values(series_to_normalize, normalization_type)
        final_data[feature] = values
        mean_or_min_to_return[feature] = mean_or_min
        std_or_max_to_return[feature] = std_or_max

    rest_of_data = all_relevant_synop_data.drop(features, axis=1)
    return pd.concat([final_data, rest_of_data], axis=1,
                     join='inner'), mean_or_min_to_return, std_or_max_to_return

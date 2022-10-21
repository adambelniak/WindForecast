import datetime
import os
from pathlib import Path
from typing import Union, Tuple

import numpy as np
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
        features = modify_feature_names_after_periodic_reduction(features)
    if norm:
        data[features], mean_or_min, std_or_max = get_normalization_values(data[features].values, normalization_type)
        return data, mean_or_min, std_or_max

    return data


def modify_feature_names_after_periodic_reduction(features: list):
    new_features = features
    for feature in SYNOP_PERIODIC_FEATURES:
        new_features.remove(feature['column'][1])
        new_features.append(f'{feature["column"][1]}-sin')
        new_features.append(f'{feature["column"][1]}-cos')
    return new_features

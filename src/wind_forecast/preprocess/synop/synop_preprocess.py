import datetime
import os
from typing import Dict

import numpy as np
from pathlib import Path
import pandas as pd

from wind_forecast.util.common_util import NormalizationType


def get_normalization_values(data: pd.DataFrame, normalization_type: NormalizationType = NormalizationType.STANDARD):
    if normalization_type == NormalizationType.STANDARD:
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        return (data - data_mean) / data_std, data_mean, data_std
    else:
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        return (data - data_min) / (data_max - data_min), data_min, data_max


def split_features_into_arrays(data, train_split, past_len, future_offset, y_column_name="velocity"):
    train_data = data.loc[:train_split - 1]

    start = past_len + future_offset
    end = start + train_split

    x_data = train_data.values
    y_data = data.iloc[start: end][[y_column_name]]

    return x_data, y_data


def prepare_synop_dataset(synop_file_name, features, norm=True, dataset_dir=os.path.join(Path(__file__).parent, 'synop_data'),
                          from_year=2001, to_year=2021, normalization_type: NormalizationType = NormalizationType.STANDARD):
    synop_file_path = os.path.join(dataset_dir, synop_file_name)
    if not os.path.exists(synop_file_path):
        raise Exception(f"Dataset not found. Looked for {synop_file_path}")

    data = pd.read_csv(synop_file_path, usecols=features + ['year', 'month', 'day', 'hour'])

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])

    first_date = datetime.datetime(year=from_year, month=1, day=1)
    last_date = datetime.datetime(year=to_year, month=1, day=1)

    data = data[(data['date'] >= first_date) & (data['date'] < last_date)]

    if norm:
        data[features], mean_or_min, std_or_max = get_normalization_values(data[features].values, normalization_type)
        return data, mean_or_min, std_or_max

    return data


def resolve_synop_data(all_synop_data: pd.DataFrame, synop_data_indices: [int], length_of_sequence):
    # Bear in mind that synop_data_indices are indices of FIRST synop in the sequence. Not all synop data exist in synop_data_indices because of that fact.
    all_indices = set([item for sublist in [[index + frame for frame in range(0, length_of_sequence)] for index in synop_data_indices]
         for item in sublist])
    return all_synop_data.take(list(all_indices))


def normalize_synop_data(all_synop_data: pd.DataFrame, synop_data_indices: [int], features: [str], length_of_sequence: int, target_param: str, normalization_type: NormalizationType = NormalizationType.STANDARD, periodic_features: [Dict] = None):
    all_relevant_synop_data = resolve_synop_data(all_synop_data, synop_data_indices, length_of_sequence)
    periodic_features_names = [x['column'][1] for x in periodic_features]
    final_data = pd.DataFrame()
    mean_or_min_to_return = 0
    std_or_max_to_return = 0
    for feature in features:
        series_to_normalize = all_relevant_synop_data[feature]
        if periodic_features is not None and feature in periodic_features_names:
            periodic_feature = [f for f in periodic_features if f['column'][1] == feature][0]
            min = periodic_feature['min']
            max = periodic_feature['max']
            final_data[feature] = np.sin(((series_to_normalize.values - min) / (max - min)).astype(np.float64) * 2 * np.pi).tolist()
            if target_param == feature:
                mean_or_min_to_return = min
                std_or_max_to_return = max
        else:
            values, mean_or_min, std_or_max = get_normalization_values(series_to_normalize.values, normalization_type)
            final_data[feature] = values
            if target_param == feature:
                mean_or_min_to_return = mean_or_min
                std_or_max_to_return = std_or_max

    rest_of_data = all_relevant_synop_data.drop(features, axis=1)
    return pd.concat([final_data, rest_of_data], axis=1, join='inner'), mean_or_min_to_return, std_or_max_to_return


def filter_for_dates(dataset, init_date, end_date):
    return dataset[(dataset["date"] > init_date) & (dataset["date"] < end_date)]

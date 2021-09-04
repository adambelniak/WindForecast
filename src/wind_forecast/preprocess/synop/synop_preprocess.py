import datetime
import os
from pathlib import Path

import pandas as pd

from wind_forecast.util.utils import NormalizationType


def normalize(data: pd.DataFrame, normalization_type: NormalizationType = NormalizationType.STANDARD):
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

    first_date = datetime.datetime(year=from_year, month=12, day=1)
    last_date = datetime.datetime(year=to_year, month=1, day=1)

    data = data[(data['date'] >= first_date) & (data['date'] < last_date)]

    if norm:
        data[features], mean_or_min, std_or_max = normalize(data[features].values, normalization_type)
        return data, mean_or_min, std_or_max

    return data, 0, 0


def filter_for_dates(dataset, init_date, end_date):
    return dataset[(dataset["date"] > init_date) & (dataset["date"] < end_date)]

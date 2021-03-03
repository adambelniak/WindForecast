import os
from pathlib import Path

import pandas as pd

from preprocess.synop import consts

DATASETS_DIRECTORY = os.path.join(Path(__file__).parent, "..", "..", "datasets", "synop")


def normalize(data):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return (data - data_mean) / data_std


def split_features_into_arrays(data, train_split, past_len, future_offset, y_column_name="velocity"):
    train_data = data.loc[:train_split - 1]

    start = past_len + future_offset
    end = start + train_split

    x_data = train_data.values
    y_data = data.iloc[start: end][[y_column_name]]

    return x_data, y_data


def prepare_synop_dataset(file_path, features, norm=True):
    synop_file_path = os.path.join(DATASETS_DIRECTORY, file_path)
    if not os.path.exists(synop_file_path):
        raise Exception(f"Dataset not fount. Looked for {synop_file_path}")

    data = pd.read_csv(synop_file_path, usecols=features + ['year', 'month', 'day', 'hour'])

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    if norm:
        data[features] = normalize(data[features].values)

    return data


def filter_for_dates(dataset, init_date, end_date):
    return dataset[(dataset["date"] > init_date) & (dataset["date"] < end_date)]


if __name__ == '__main__':
    prepare_synop_dataset("synop_data/135_data.csv", [consts.VELOCITY_COLUMN])

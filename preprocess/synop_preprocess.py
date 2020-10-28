import pandas as pd
from preprocess.fetch_synop_data import FEATURES


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


def prepare_synop_dataset(file_path):
    data = pd.read_csv(file_path)

    data["date"] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data[FEATURES] = normalize(data[FEATURES].values)
    print(data.head())

    return data


if __name__ == '__main__':
    prepare_synop_dataset("synop_data/135_data.csv")

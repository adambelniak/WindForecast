import os
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from preprocess.synop_preprocess import normalize

CREATED_AT_COLUMN_NAME = "created_at"

def prepare_gfs_data(dir):
    gfs_data = {}

    for file_name in os.listdir(dir):
        single_gfs = pd.read_csv(os.path.join(dir, file_name))

        date_forecast_re = re.search(r'^([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])-([0|1][0-9]|2[0-4])Z).csv*',
                                     file_name)
        if not date_forecast_re:
            raise Exception("Invalid file name format - except 'YYYY-MM-DD-HHZ.csv'")

        date_forecast = date_forecast_re.group(1)
        gfs_data[date_forecast] = single_gfs
    return gfs_data


def filter_desired_time_stride(data: dict, time_stride: int):
    if time_stride % 3:
        raise Exception("Time stride must be number divisible by 3")

    single_gfs_frame = next(iter(data.values()))

    data_for_single_time_stride = pd.DataFrame(columns=single_gfs_frame.columns)
    for gfs_time_key in sorted(data.keys()):
        single_gfs = data[gfs_time_key]
        date_time = datetime.strptime(gfs_time_key, '%Y-%m-%d-%HZ') + timedelta(hours=time_stride)

        single_gfs['date'] = pd.to_datetime(single_gfs['date'])
        filtered = single_gfs[single_gfs['date'] == date_time]
        filtered.columns = single_gfs_frame.columns

        data_for_single_time_stride = data_for_single_time_stride.append(filtered, ignore_index=True)

    return data_for_single_time_stride


def show_raw_for_desired_time_stride(data: dict, time_stride: int, feature_keys: list):

    data_for_single_time_stride = filter_desired_time_stride(data, time_stride)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")


    for i in range(len(feature_keys)):
        key = feature_keys[i]
        t_data = data_for_single_time_stride[key]
        t_data.index = data_for_single_time_stride["date"]
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            title="{}".format(feature_keys[i]),
            rot=25,
        )
        ax.legend([feature_keys[i]])
    plt.show()


def show_heatmap_for_desired_time_stride(data, time_stride, columns):
    data_for_single_time_stride = filter_desired_time_stride(data, time_stride)
    data_for_single_time_stride = data_for_single_time_stride.drop('date', 1)

    plt.matshow(data_for_single_time_stride.corr())
    plt.xticks(range(data_for_single_time_stride.shape[1]), data_for_single_time_stride.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data_for_single_time_stride.shape[1]), data_for_single_time_stride.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()

def filter_time_stride_with_origin_date(data: dict, time_stride: int):
    if time_stride % 3:
        raise Exception("Time stride must be number divisible by 3")

    single_gfs_frame = next(iter(data.values()))
    data_for_single_time_stride = pd.DataFrame(columns=single_gfs_frame.columns)
    for gfs_time_key in sorted(data.keys()):
        single_gfs = data[gfs_time_key]
        date_time = datetime.strptime(gfs_time_key, '%Y-%m-%d-%HZ') + timedelta(hours=time_stride)

        single_gfs['date'] = pd.to_datetime(single_gfs['date'])
        filtered = single_gfs[single_gfs['date'] == date_time]
        filtered.columns = single_gfs_frame.columns
        filtered[CREATED_AT_COLUMN_NAME] = datetime.strptime(gfs_time_key, '%Y-%m-%d-%HZ')

        data_for_single_time_stride = data_for_single_time_stride.append(filtered, ignore_index=True)

    return data_for_single_time_stride


def process_and_plot(dir, time_stride, index_column):
    data = prepare_gfs_data(dir)
    single_gfs_frame = next(iter(data.values()))
    columns = [column for column in single_gfs_frame.columns if column != index_column]

    show_raw_for_desired_time_stride(data, time_stride, columns)
    show_heatmap_for_desired_time_stride(data, time_stride, columns)


def normalize_data_without_dates(data, index_column):
    columns_withou_dates = [column for column in data.columns if column not in [index_column, CREATED_AT_COLUMN_NAME]]
    data[columns_withou_dates] = normalize(data[columns_withou_dates].values)
    return data


def prepare_gfs_dataset(dir: str, index_column: str, time_stride=12, train_split_factor=0.75):
    data = prepare_gfs_data(dir)
    data_for_single_time_stride = filter_time_stride_with_origin_date(data, time_stride)
    data_for_single_time_stride = normalize_data_without_dates(data_for_single_time_stride, index_column)

    return data_for_single_time_stride


if __name__ == "__main__":
    index_column = "date"
    prepare_gfs_dataset("./wind", index_column, 12)

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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20), dpi=80, facecolor="w", edgecolor="k")

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
    plt.xticks(range(data_for_single_time_stride.shape[1]), data_for_single_time_stride.columns, fontsize=14,
               rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data_for_single_time_stride.shape[1]), data_for_single_time_stride.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


def filter_desired_time_stride_with_origin_date(data: dict, time_stride: int, number_future_samples: int):
    """
    This method return GFS prediction for provided point of time.
    :param data: dictionary contains gfs by creation time
    :param time_stride: in hours - specify which gfs prediction should be considered; interested time = created_at + time_stride
    :return:
    """
    if time_stride % 3:
        raise Exception("Time stride must be number divisible by 3")

    single_gfs_frame = next(iter(data.values()))
    data_for_single_time_stride = pd.DataFrame(columns=single_gfs_frame.columns)
    for gfs_time_key in sorted(data.keys()):
        single_gfs = data[gfs_time_key]
        start_gfs_prediction_time = datetime.strptime(gfs_time_key, '%Y-%m-%d-%HZ') + timedelta(hours=time_stride)
        end_gfs_prediction_time = datetime.strptime(gfs_time_key, '%Y-%m-%d-%HZ') + timedelta(
            hours=time_stride + number_future_samples)

        single_gfs['date'] = pd.to_datetime(single_gfs['date'])
        filtered = single_gfs[
            (single_gfs['date'] >= start_gfs_prediction_time) & (single_gfs['date'] <= end_gfs_prediction_time)]
        filtered.columns = single_gfs_frame.columns
        filtered[CREATED_AT_COLUMN_NAME] = datetime.strptime(gfs_time_key, '%Y-%m-%d-%HZ')

        data_for_single_time_stride = data_for_single_time_stride.append(filtered, ignore_index=True)

    return data_for_single_time_stride


def get_gfs_time_laps(gfs_creation_time):
    creation_time = sorted(gfs_creation_time)[:2]
    return datetime.strptime(creation_time[1], '%Y-%m-%d-%HZ') - datetime.strptime(creation_time[0], '%Y-%m-%d-%HZ')


def filter_desired_time_with_future(data: dict, time_stride: int, number_future_samples: int):
    """
    This method return GFS predictions for provided point of time together with next samples.
    :param data: dictionary contains gfs by creation time
    :param time_stride: in hours - specify which gfs prediction should be considered; interested time = created_at + time_stride
    :return:
    """
    gfs_file_time_laps = get_gfs_time_laps(data.keys())
    gfs_file_time_laps_hours = gfs_file_time_laps.seconds // 3600
    if gfs_file_time_laps_hours < number_future_samples:
        raise Exception("Hours of future samples cannot be grater than difference of gfs creation")
    data_for_single_time_stride = filter_desired_time_stride_with_origin_date(data, time_stride, number_future_samples)


def create_gfs_sequence(data: dict, start_sequence: int, end_sequence: int, past_samples_length: int,
                        future_samples_length: int):
    """

    :param data: dictionary contains gfs data from files
    :param start_sequence: index of start sample from which the sequences will be generated
    :param end_sequence: index of end sample until which the sequences will be generated
    :param past_samples_length: number of samples before point time, to be considered during sequence generating
    :param future_samples_length: number of samples after point time, to be considered during sequence generating
    :return:
    """
    single_gfs_data_frame = next(iter(data.values()))

    if end_sequence <= start_sequence:
        raise Exception("Start bias time cannot be greater than end bias")
    if start_sequence - past_samples_length < 0:
        raise Exception("Time window for the Past cannot be less than 0")
    if end_sequence + future_samples_length > single_gfs_data_frame.shape[0]:
        raise Exception("Time window for the Future cannot be greater than count samples in single GFS")

    gfs_sequences = {}
    for gfs_time_key in sorted(data.keys()):
        single_gfs = data[gfs_time_key]
        start = start_sequence - past_samples_length
        end = end_sequence + future_samples_length
        single_gfs = single_gfs.iloc[start:end]
        single_gfs['date'] = pd.to_datetime(single_gfs['date'])

        gfs_sequences[gfs_time_key] = single_gfs
    return gfs_sequences


def wind_transformation(gfs_data_frames):
    transformed = {}
    for gfs_time_key in sorted(gfs_data_frames.keys()):
        single_gfs = gfs_data_frames[gfs_time_key]
        single_gfs["gfs_velocity"] = np.sqrt(single_gfs["U-wind, m/s"] ** 2 + single_gfs["V-wind, m/s"] ** 2)
        single_gfs = single_gfs.drop(["U-wind, m/s", "V-wind, m/s"], axis=1)
        transformed[gfs_time_key] = single_gfs
    return transformed


def process_and_plot(dir, time_stride, index_column):
    data = prepare_gfs_data(dir)
    data = wind_transformation(data)
    single_gfs_frame = next(iter(data.values()))
    columns = [column for column in single_gfs_frame.columns if column != index_column]

    show_raw_for_desired_time_stride(data, time_stride, columns)
    show_heatmap_for_desired_time_stride(data, time_stride, columns)


def normalize_data_without_dates(data, index_column):
    columns_withou_dates = [column for column in data.columns if column not in [index_column, CREATED_AT_COLUMN_NAME]]
    data[columns_withou_dates] = normalize(data[columns_withou_dates].values)
    return data


def prepare_gfs_dataset_for_single_point_time(dir: str, index_column: str, time_stride=12):
    data = prepare_gfs_data(dir)
    data = wind_transformation(data)
    data_for_single_time_stride = filter_desired_time_stride_with_origin_date(data, time_stride, 0)
    # data_for_single_time_stride = normalize_data_without_dates(data_for_single_time_stride, index_column)

    return data_for_single_time_stride


def get_single_gfs_time_dif(gfs):
    gfs = pd.to_datetime(gfs['date'])
    return gfs.at[1] - gfs.at[0]


def get_single_gfs_start_time(gfs_data: dict):
    created_at_key = next(iter(gfs_data.keys()))
    gfs = gfs_data[created_at_key]
    gfs = pd.to_datetime(gfs['date'])

    return gfs.at[0] - datetime.strptime(created_at_key, '%Y-%m-%d-%HZ')


def convert_hours_to_index(gfs_hour_diff: int, gfs_start_at: int, interested_hour):
    return (interested_hour - gfs_start_at) // gfs_hour_diff


def prepare_gfs_sequence_dataset(dir: str, start_sequence_hour: int, end_sequence_hour: int,
                                 past_samples_length_hour: int,
                                 future_samples_length_hour: int):
    data = prepare_gfs_data(dir)
    gfs_hour_diff = get_single_gfs_time_dif(next(iter(data.values()))).seconds // 3600
    gfs_start_at = get_single_gfs_start_time(data).seconds // 3600

    start_sequence_index = convert_hours_to_index(gfs_hour_diff, gfs_start_at, start_sequence_hour)
    end_sequence_index = convert_hours_to_index(gfs_hour_diff, gfs_start_at, end_sequence_hour) + 1
    past_samples_length = past_samples_length_hour // gfs_hour_diff
    future_samples_length = future_samples_length_hour // gfs_hour_diff

    gfs_sequences = create_gfs_sequence(data, start_sequence_index, end_sequence_index, past_samples_length,
                                        future_samples_length)

    return gfs_sequences, gfs_hour_diff


if __name__ == "__main__":
    index_column = "date"
    # data = prepare_gfs_data("./wind")

    data = process_and_plot('./wind_and_temp', 12, 'date')
    print(data.head)

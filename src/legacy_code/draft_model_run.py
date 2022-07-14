from datetime import timedelta

from wind_forecast.consts import CREATED_AT_COLUMN_NAME
from legacy_code.models.gfs_model import create_model
from legacy_code.preprocess.gfs.gfs_preprocess_csv import prepare_gfs_dataset_for_single_point_time
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
import numpy as np

AUXILIARY_COLUMNS = ["10m U-wind, m/s", "10m V-wind, m/s", "Gusts, m/s"]


def get_gfs_time_laps(gfs_data):
    """
    Here it's used 'array' instead of 'values' because values convert series to dateTime64 from numpy
    :param gfs_data:
    :return:
    """
    gfs_dates = gfs_data['created_at']
    return gfs_dates.array[1] - gfs_dates.array[0]


def create_sequence(data, gfs_data, past_len, future_offset):
    train_size = data.shape[0]
    train_synop_input = []
    train_synop_label = []
    gfs_input = []
    gfs_columns_without_dates = [column for column in gfs_data.columns if
                                 column not in ['date', CREATED_AT_COLUMN_NAME]]
    synop_columns_without_dates = [column for column in data.columns if column not in ['date', 'year', 'day', 'month']]
    gfs_time_diff = get_gfs_time_laps(gfs_data)
    gfs_index = gfs_time_diff.seconds // 3600
    for single_stride in range(train_size // gfs_index - 4):
        past_synop_data = data.iloc[gfs_index * single_stride:(single_stride + 1) * gfs_index + past_len][
            synop_columns_without_dates].values
        gfs_prediction = gfs_data.iloc[single_stride]
        gfs_prediction = gfs_prediction[gfs_columns_without_dates].values

        # wind_velocity = np.sqrt(gfs_prediction["U-wind, m/s"] ** 2 + gfs_prediction["V-wind, m/s"] ** 2)
        future_index = gfs_index * (single_stride + 1) + future_offset
        label = data.iloc[future_index][['temperature']].values.flatten()
        train_synop_input.append(past_synop_data.astype(np.float32))
        train_synop_label.append(label.astype(np.float32))
        gfs_input.append(gfs_prediction.astype(np.float32))

    return np.array(train_synop_input), np.array(gfs_input), np.array(train_synop_label)


def prepare_data(past_len=12, future_offset=12, train_split_factor=0.75):
    features = []
    dataset_train, _, _ = prepare_synop_dataset("preprocess/synop_data/135_data.csv", "")
    gfs_dataset = prepare_gfs_dataset_for_single_point_time("preprocess/wind_and_temp", 'date', future_offset)

    least_recent_date = gfs_dataset["date"].min()
    dataset_train = dataset_train[dataset_train["date"] >= least_recent_date - timedelta(hours=past_len)]
    gfs_dataset = gfs_dataset[gfs_dataset["date"] > least_recent_date + timedelta(hours=future_offset)]
    gfs_dataset = gfs_dataset.sort_values(by=["date"])

    train_synop_input, gfs_input, train_synop_label = create_sequence(dataset_train, gfs_dataset, past_len,
                                                                      future_offset)
    train_size = int(train_synop_input.shape[0] * train_split_factor)

    x_train = [train_synop_input[:train_size], gfs_input[:train_size]]
    y_train = train_synop_label[:train_size]
    x_valid = [train_synop_input[train_size:], gfs_input[train_size:]]
    y_valid = train_synop_label[train_size:]

    return x_train, y_train, x_valid, y_valid


def run_model():
    x_train, y_train, x_valid, y_valid = prepare_data()

    model = create_model(x_train[0], x_train[1], 0.001)
    history = model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_valid, y_valid), )


if __name__ == '__main__':
    run_model()

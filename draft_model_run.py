from datetime import timedelta

from models.gfs_model import create_model
from preprocess.gfs_preprocess import prepare_gfs_dataset, CREATED_AT_COLUMN_NAME
from preprocess.synop_preprocess import prepare_synop_dataset, split_features_into_arrays
import pandas as pd
import numpy as np

AUXILIARY_COLUMNS = ["10m U-wind, m/s", "10m V-wind, m/s", "Gusts, m/s"]


def create_sequence(data, gfs_data, past_len, future_offset):
    train_size = data.shape[0]
    train_synop_input = []
    train_synop_label = []
    gfs_input = []
    gfs_columns_without_dates = [column for column in gfs_data.columns if column not in ['date', CREATED_AT_COLUMN_NAME]]
    synop_columns_withou_dates = [column for column in data.columns if column != 'date']

    for single_stride in range(train_size // past_len - 2):
        # print((past_len + 1) * single_stride)
        past_synop_data = data.iloc[past_len * single_stride:(single_stride + 1) * past_len][synop_columns_withou_dates].values
        gfs_prediction = gfs_data.iloc[single_stride]
        gfs_prediction = gfs_prediction[gfs_columns_without_dates].values
        future_index = past_len * (single_stride + 1) + future_offset
        # print(future_index)
        label = data.iloc[future_index][['velocity']].values.flatten()

        train_synop_input.append(past_synop_data.astype(np.float32))
        train_synop_label.append(label.astype(np.float32))
        gfs_input.append(gfs_prediction.astype(np.float32))

    return np.array(train_synop_input), np.array(gfs_input), np.array(train_synop_label)


def prepare_data(past_len=12, future_offset=12, train_split_factor=0.75):
    dataset_train = prepare_synop_dataset("preprocess/synop_data/135_data.csv")
    gfs_dataset = prepare_gfs_dataset("preprocess/wind", 'date', future_offset)

    train_split = int(len(gfs_dataset) * train_split_factor)

    start = past_len + future_offset
    end = start + train_split

    least_recent_date = gfs_dataset["date"].min()
    dataset_train = dataset_train[dataset_train["date"] > least_recent_date - timedelta(hours=past_len)]
    gfs_dataset = gfs_dataset[gfs_dataset["date"] > least_recent_date + timedelta(hours=future_offset)]

    train_synop_input, gfs_input, train_synop_label = create_sequence(dataset_train, gfs_dataset, past_len,
                                                                      future_offset)

    print(train_synop_input.shape)
    model = create_model(train_synop_input, gfs_input, 0.001)
    #
    history = model.fit([train_synop_input, gfs_input], [train_synop_label], epochs=300, batch_size=32)

    step = 1
    past_len = 20
    train_split_factor = 0.75
    sequence_length = int(past_len / step)
    train_split = int(len(dataset_train) * train_split_factor)
    future_offset = 12

    # x_train, y_train = split_features_into_arrays(dataset_train, train_split, past_len, future_offset)


def run_model():
    dataset_train = prepare_synop_dataset("preprocess/synop_data/135_data.csv")
    step = 1
    past_len = 20

    sequence_length = int(past_len / step)

    for batch in dataset_train.take(1):
        inputs, targets = batch
    model = create_model(inputs, 0.001)

    history = model.fit(
        dataset_train,
        epochs=10,
    )


if __name__ == '__main__':
    prepare_data()

import pandas as pd
import tensorflow.keras as keras

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


def prepare_synop_dataset(file_path, train_split_factor=0.75, step=1, batch_size=32, past_len=20, label_len=1,
                          future_offset=10):
    data = pd.read_csv(file_path)
    train_split = int(len(data) * train_split_factor)

    print(data.head())
    data[FEATURES] = normalize(data.values)

    sequence_length = int(past_len / step)

    x_train, y_train = split_features_into_arrays(data, train_split, past_len, future_offset)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        # label_len,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )
    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)


if __name__ == '__main__':
    prepare_synop_dataset("../synop_data/135_data.csv")

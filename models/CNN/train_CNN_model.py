import argparse
import sys
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import numpy as np

sys.path.insert(1, '../..')

from models.common import GFS_PARAMETERS, plot_history
from preprocess.synop.consts import TEMPERATURE, VELOCITY_COLUMN
from models.CNN.CNN_model import create_model


from preprocess.gfs.gfs_preprocess_netCDF import get_forecasts_for_year_offset_param_from_npy_file
from preprocess.synop.synop_preprocess import prepare_synop_dataset, filter_for_dates


def prepare_target_attribute_dataset(synop_data_file, target_attribute, init_date, end_date):
    dataset = prepare_synop_dataset(synop_data_file, [target_attribute])
    return filter_for_dates(dataset, init_date, end_date)


def prepare_labels(target_attribute, forecast_offset_to_train, years, synop_dataset_path):
    labels = []
    for year in years:
        if year == 2015:
            init_date = datetime(2015, 1, 15)
        else:
            init_date = datetime(year, 1, 1)
        end_date = datetime(init_date.year + 1, 1, 1)

        synop_data = prepare_target_attribute_dataset(synop_dataset_path, target_attribute, init_date, end_date)

        date = init_date
        while date < end_date:
            for index, run in enumerate(['00', '06', '12', '18']):
                forecast_date = date + timedelta(hours=index * 6 + forecast_offset_to_train)
                label = synop_data[synop_data["date"] == forecast_date][target_attribute].values[0]
                labels.append(np.array([label]))
            date = date + timedelta(days=1)

    return np.array(labels)


def prepare_forecasts(features, forecast_offset_to_train, years, gfs_dataset_dir):
    forecasts = []

    for feature in features:
        forecasts_for_feature = None
        for year in years:
            np_arr = get_forecasts_for_year_offset_param_from_npy_file(year, feature, forecast_offset_to_train, gfs_dataset_dir)
            if forecasts_for_feature is None:
                forecasts_for_feature = np_arr
            else:
                forecasts_for_feature = np.concatenate((forecasts_for_feature, np_arr))
        forecasts.append(forecasts_for_feature)

    arr = np.array(forecasts)
    return np.einsum('klij->lijk', arr)


def prepare_training_data(features, target_attribute, forecast_offset_to_train, years, gfs_dataset_dir, synop_dataset_path):
    labels = prepare_labels(target_attribute, forecast_offset_to_train, years, synop_dataset_path)
    forecasts = prepare_forecasts(features, forecast_offset_to_train, years, gfs_dataset_dir)

    return forecasts, labels


def train_model(training_data):
    print(training_data[0].shape)
    x_train, x_valid, y_train, y_valid = train_test_split(training_data[0], training_data[1], test_size=0.2, shuffle=True)
    model = create_model(training_data[0][0].shape)
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_data=(x_valid, y_valid))
    plot_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--years', help='Years of GFS forecasts to take into account.', nargs="+", required=True, type=int)
    parser.add_argument('--gfs_dataset_dir', help='Directory with dataset of gfs forecasts.', type=str, default="D:\\WindForecast\\output_np")
    parser.add_argument('--synop_dataset_path', help='Directory with dataset of gfs forecasts.', type=str, default="../../datasets/synop/KOZIENICE_488_data.csv")

    args = parser.parse_args()

    training, labels = prepare_training_data(GFS_PARAMETERS, VELOCITY_COLUMN[1], 3, args.years, args.gfs_dataset_dir, args.synop_dataset_path)

    train_model((training, labels))


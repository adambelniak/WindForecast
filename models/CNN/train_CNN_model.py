import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm
import numpy as np
sys.path.insert(1, '../..')
from CNN_model import create_model
from preprocess.synop import consts


from preprocess.gfs.gfs_preprocess_netCDF import get_forecasts_for_date_offsets_and_params
from preprocess.synop.synop_preprocess import prepare_synop_dataset, filter_for_dates


def prepare_target_attribute_dataset(synop_data_file, target_attribute, init_date, end_date):
    dataset = prepare_synop_dataset(synop_data_file, [target_attribute])
    return filter_for_dates(dataset, init_date, end_date)


def prepare_training_data(features, target_attribute, forecast_offset_to_train, init_date, end_date, synop_data):
    forecasts, labels = [], []
    date = init_date
    delta = end_date - init_date
    for i in tqdm(range(delta.days)):
        for index, run in enumerate(['00', '06', '12', '18']):
            forecast = get_forecasts_for_date_offsets_and_params(date, run, forecast_offset_to_train, forecast_offset_to_train, features, 56, 48, 13, 26)
            forecast_date = date + timedelta(hours=index * 6 + forecast_offset_to_train)
            label = synop_data[synop_data["date"] == forecast_date][target_attribute].values[0]
            forecasts.append(forecast)
            labels.append(label)
        date = date + timedelta(days=1)

    np.save(Path(os.path.join(__file__, "..", "..", "..", "datasets", "gfs", "2015-2016-U_GRD-ISBL_925-f003.npy")).resolve(), forecasts)

    return forecasts, labels


def train_model(synop_data_file: str, features: [(str, str)], target_attribute, forecast_offset_to_train, init_date=None, end_date=None):
    init_date = init_date if init_date is not None else datetime(2015, 1, 1)
    end_date = end_date if end_date is not None else datetime(2021, 1, 1)

    synop_data = prepare_target_attribute_dataset(synop_data_file, target_attribute, init_date, end_date)
    training_data = prepare_training_data(features, target_attribute, forecast_offset_to_train, init_date, end_date, synop_data)

    print(training_data[0][0].shape)
    # model = create_model(training_data[0][0].shape)


if __name__ == "__main__":
    # train_model("135_data.csv", GFS_PARAMETERS, consts.TEMPERATURE[1], 3, datetime(2015, 1, 15), datetime(2016, 1, 1))
    with open('D:\\WindForecast\\2015-2016-U_GRD-ISBL_925-f003.npy', 'rb') as f:
        a = np.load(f)
    print(a.shape)

import sys
from datetime import datetime, timedelta

import numpy as np
import os

from util.utils import utc_to_local

sys.path.insert(1, '../..')


from preprocess.gfs.gfs_preprocess_netCDF import get_forecasts_for_year_offset_param_from_npy_file
from preprocess.synop.synop_preprocess import prepare_synop_dataset, filter_for_dates


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
                forecast_date = utc_to_local(date + timedelta(hours=index * 6 + forecast_offset_to_train))
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


def prepare_training_data(features: [(str, str)], target_attribute: str, forecast_offset_to_train: int, years: [int],
                          gfs_dataset_dir: str, synop_dataset_path: str):
    labels = prepare_labels(target_attribute, forecast_offset_to_train, years, synop_dataset_path)
    forecasts = prepare_forecasts(features, forecast_offset_to_train, years, gfs_dataset_dir)

    return forecasts, labels


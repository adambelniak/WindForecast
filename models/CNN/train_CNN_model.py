import sys
from datetime import datetime

sys.path.insert(1, '../..')

from preprocess.gfs.gfs_preprocess_netCDF import get_forecasts_for_date_range_all_runs_specified_offsets_and_params
from preprocess.synop.synop_preprocess import prepare_synop_dataset, filter_for_dates


def prepare_target_attribute_dataset(synop_data_file, target_attribute, init_date, end_date):
    dataset = prepare_synop_dataset(synop_data_file, [target_attribute])
    return filter_for_dates(dataset, init_date, end_date)


def prepare_gfs_training_data(features, forecast_offset_to_train, init_date, end_date):
    return get_forecasts_for_date_range_all_runs_specified_offsets_and_params(init_date, end_date, forecast_offset_to_train,
                                                                              forecast_offset_to_train + 3, features, 56, 48, 13, 26)


def train_model(synop_data_file: str, features: [(str, str)], target_attribute, forecast_offset_to_train, init_date=None, end_date=None):
    init_date = init_date if init_date is not None else datetime(2015, 1, 1)
    end_date = end_date if end_date is not None else datetime(2021, 1, 1)

    labels = prepare_target_attribute_dataset(synop_data_file, target_attribute, init_date, end_date)
    training_data = prepare_gfs_training_data(features, forecast_offset_to_train, init_date, end_date)
    # TODO match training data with labels, reshape training dataset
    batch_size = 32
    # width = training_data.shape[]


if __name__ == "__main__":
    print(prepare_gfs_training_data([("T CDC", "LCY_0")], 3, datetime(2015, 1, 1), datetime(2019, 1, 1)))

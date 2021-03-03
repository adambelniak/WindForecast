import os
import argparse
import numpy as np
import tqdm

from pathlib import Path
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.utils import shuffle

from models.common import plot_history, convert_wind
from models.dense.dense_model import create_model
from preprocess.synop import consts
from preprocess.synop.synop_preprocess import prepare_synop_dataset
from preprocess.gfs.gfs_preprocess_csv import prepare_gfs_data

GFS_CSV_DIR = os.path.join(Path(__file__), "..", "..", 'gfs-archive-0-25/gfs_processor/output_csvs/54_6-18_8')


def add_hours_based_on_time_shift(date):
    if date.month > 10 or date.month < 4:
        return date + timedelta(hours=1)
    else:
        return date + timedelta(hours=2)


def prepare_single_hour_data(single_gfs, synop_dataset, parameter, specific_hour):
    condition = single_gfs.date.str.contains(specific_hour)
    single_gfs_spec_hour = single_gfs[condition]
    if len(single_gfs_spec_hour.index) == 1:
        # don't take first two rows, because clouds are not correct
        if single_gfs.index[condition] in [0, 1]:
            return None, None
        else:
            single_gfs_spec_hour = single_gfs_spec_hour.iloc[0]
    elif len(single_gfs_spec_hour.index) == 0:
        return None, None
    else:
        single_gfs_spec_hour = single_gfs_spec_hour.iloc[1]
    if len(single_gfs_spec_hour.index) != 9:
        return None, None
    single_gfs_spec_hour['velocity'] = convert_wind(single_gfs_spec_hour, 'U GRD_HTGL:10', 'V GRD_HTGL:10')
    date = datetime.strptime(single_gfs_spec_hour['date'], '%Y-%m-%dT%H:%M:%S')
    date = add_hours_based_on_time_shift(date)
    single_gfs_spec_hour.drop(['date', 'U GRD_HTGL:10', 'V GRD_HTGL:10'], inplace=True)

    return single_gfs_spec_hour.to_numpy(), np.array([synop_dataset.loc[synop_dataset['date'] == date].drop('date', axis=1).iloc[0][parameter]])


def prepare_gfs_data_for_forecast_frame(single_gfs, synop_dataset, parameter, forecast_frame: int):
    if len(single_gfs.index) < forecast_frame:
        return None, None
    single_gfs = single_gfs[2:forecast_frame]
    gfs_result = []
    labels_result = []
    for i in range(2, forecast_frame):
        velocity = convert_wind(single_gfs.loc[i], 'U GRD_HTGL:10', 'V GRD_HTGL:10')
        date = datetime.strptime(single_gfs.loc[i]['date'], '%Y-%m-%dT%H:%M:%S')
        date = add_hours_based_on_time_shift(date)
        vals_in_gfs = single_gfs.drop(['date', 'U GRD_HTGL:10', 'V GRD_HTGL:10'], axis=1).loc[i]
        vals_in_gfs['velocity'] = velocity
        if len(vals_in_gfs.index) != 7:
            continue
        gfs_result.append(vals_in_gfs.to_numpy())
        labels_result.append(np.array([synop_dataset.loc[synop_dataset['date'] == date].drop('date', axis=1).iloc[0][parameter]]))

    return gfs_result, labels_result


def prepare_data(gfs_data_dir: str, synop_data_path: str, parameter: str, specific_hour: str, forecast_frame: int):
    print("Creating data set")

    synop_dataset = prepare_synop_dataset(synop_data_path, [parameter], False)

    if parameter not in synop_dataset.columns:
        raise Exception("Parameter {} is not available in synop dataset.".format(parameter))

    gfs_dataset_by_forecast_date = prepare_gfs_data(gfs_data_dir)

    gfs_dataset_input, dataset_labels = [], []
    for key in tqdm.tqdm(gfs_dataset_by_forecast_date.keys()):
        single_gfs = gfs_dataset_by_forecast_date[key]

        if specific_hour != '':  # train only for the specific hour of the first day in forecast
            gfs_data, labels_data = prepare_single_hour_data(single_gfs, synop_dataset, parameter, specific_hour)
            if gfs_data is None:
                continue
            gfs_dataset_input.append(gfs_data)
            dataset_labels.append(labels_data)
        else:
            gfs_data, labels_data = prepare_gfs_data_for_forecast_frame(single_gfs, synop_dataset, parameter, forecast_frame)
            if gfs_data is None:
                continue
            gfs_dataset_input.extend(gfs_data)
            dataset_labels.extend(labels_data)

    print(gfs_dataset_input[:5])
    print(dataset_labels[:5])
    print("Labels max value: {}, min value: {}".format(max(dataset_labels), min(dataset_labels)))
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(gfs_dataset_input), \
           min_max_scaler.fit_transform(dataset_labels)


def train_model(**kwargs):
    dataset_input, dataset_labels = prepare_data(kwargs['gfs_data_dir'], kwargs['synop_data_csv'], kwargs['parameter'],
                                                 kwargs['spec_hour'], kwargs['forecast_frame'])

    dataset_input, dataset_labels = shuffle(dataset_input, dataset_labels)

    model = create_model()
    history = model.fit(x=dataset_input, y=dataset_labels, epochs=40, batch_size=32, validation_split=kwargs['train_split'])
    plot_history(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gfs_data_dir', help='Directory with GFS forecasts in csv format', default=GFS_CSV_DIR)
    parser.add_argument('--synop_data_csv', help='CSV with SYNOP data', default='preprocess/synop_data/135_data.csv')
    parser.add_argument('--train_split', help='Train split factor from 0 to 1', default=0.2, type=float)
    parser.add_argument('--parameter', help='Parameter to train on', default='temperature', type=str)
    parser.add_argument('--spec_hour', help='Train only for a specific hour of a first day in forecast', default='', type=str)
    parser.add_argument('--forecast_frame', help='How many steps of GFS forecast should be taken into account. '
                                                 'Applied only if spec_hour is not specified', default=8, type=int)
    args = parser.parse_args()
    train_model(**vars(args))

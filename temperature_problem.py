from models.temperature_model import create_model
import argparse
import numpy as np
from preprocess.synop_preprocess import prepare_synop_dataset, normalize
from preprocess.gfs_preprocess import prepare_gfs_data
from util.utils import convert_wind
import tqdm
import re
from datetime import datetime, timedelta
from train_sequence_model import plot_history

GFS_CSV_DIR = 'gfs-archive-0-25/gfs_processor/output_csvs/54_6-18_8'


def prepare_data(gfs_data_dir: str, synop_data_dir: str):
    print("Creating data set")
    synop_dataset = prepare_synop_dataset(synop_data_dir).between_time('05:00', '05:00', axis='date')
    gfs_dataset_by_forecast_date = prepare_gfs_data(gfs_data_dir)

    gfs_dataset_input, dataset_labels = [], []
    for key in tqdm.tqdm(gfs_dataset_by_forecast_date.keys()):
        single_gfs = gfs_dataset_by_forecast_date[key]
        single_gfs_date = single_gfs.loc[re.search(r'03:00:00', single_gfs['date'])]
        single_gfs_date = convert_wind(single_gfs_date, 'U GRD_HTGL:10', 'V GRD_HTGL:10')
        gfs_dataset_input.append(normalize(single_gfs_date.values))
        date = datetime.fromisoformat(single_gfs_date['date'])
        if date.month > 10 or date.month < 4:
            date = date + timedelta(hours=1)
        else:
            date = date + timedelta(hours=2)

        dataset_labels.append(synop_dataset.loc[synop_dataset['date'] == date][0]['temperature'].value)

    return np.array(gfs_dataset_input), np.array(dataset_labels)


def train_model(**kwargs):
    dataset_input, dataset_labels = prepare_data(kwargs['gfs_data_dir'], kwargs['synop_data_dir'])
    print("Dataset size: {}".format(np.shape(dataset_input)[0]))
    train_index = int(dataset_input.shape[0] * kwargs["train_split"])
    x_train = dataset_input[:train_index]
    x_valid = dataset_input[train_index:]
    y_train = dataset_labels[:train_index]
    y_valid = dataset_labels[train_index:]

    model = create_model(dataset_input.shape[1])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_valid, y_valid))
    plot_history(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gfs_data_dir', help='Directory with GFS forecasts in csv format', default=GFS_CSV_DIR)
    parser.add_argument('--synop_data_dir', help='Directory with SYNOP data', default='preprocess/synop_data/135_data.csv')
    parser.add_argument('--train_split', help='Train split factor from 0 to 1', default=0.9, type=float)
    args = parser.parse_args()
    # train_model(**vars(args))

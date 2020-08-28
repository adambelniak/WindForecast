import argparse
import os
from datetime import datetime, timedelta
import numpy as np
from preprocess.gfs_preprocess import prepare_gfs_sequence_dataset, CREATED_AT_COLUMN_NAME
from preprocess.synop_preprocess import prepare_synop_dataset
import pandas as pd
import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_oldest_gfs_date(gfs: dict):
    gfs.keys()
    gfs_dates = [datetime.strptime(date, '%Y-%m-%d-%HZ').date() for date in gfs.keys()]
    return min(gfs_dates)


def prepare_data(gfs_dir: str, synop_dir: str, start_seq: int, end_seq: int, gfs_past: int, gfs_future: int,
                 synop_length: int, train_split: int):
    synop_dataset = prepare_synop_dataset(synop_dir)
    gfs_dataset, gfs_hour_diff = prepare_gfs_sequence_dataset(gfs_dir, start_seq, end_seq, gfs_past, gfs_future)

    least_recent_date = get_oldest_gfs_date(gfs_dataset)
    synop_dataset = synop_dataset[
        synop_dataset["date"] >= pd.to_datetime(least_recent_date) - timedelta(hours=synop_length)]
    dataset_input = []
    dataset_label = []
    print("Creating data set")
    for key in tqdm.tqdm(gfs_dataset.keys()):
        gfs_creation_date = datetime.strptime(key, '%Y-%m-%d-%HZ')
        single_gfs = gfs_dataset[key]
        for hour in range(gfs_hour_diff):
            try:
                synop_index = synop_dataset.index[synop_dataset['date'] == gfs_creation_date][0]
                synop_input_index = gfs_hour_diff + synop_index - synop_length + 1
                synop_input = synop_dataset.loc[synop_input_index: synop_input_index + synop_length]

                label = synop_dataset.loc[synop_index + start_seq: synop_index + end_seq]

                synop_input = synop_input.drop(['date', 'year', 'day', 'month'], axis=1).values
                gfs_input = single_gfs.drop(['date'], axis=1).values
                label = label.drop(['date', 'year', 'day', 'month'], axis=1).values

                dataset_input.append([synop_input, gfs_input])
                dataset_label.append(label)
            except:
                pass
    return dataset_input, dataset_label


def train_model(**kwargs):
    dataset_input, dataset_label = prepare_data(**kwargs)
    print("Dataset size: {}".format(np.shape(dataset_input))[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gfs_dir', help='GFS directory', default='preprocess/wind_and_temp')
    parser.add_argument('--synop_dir', help='SYNOP directory', default='preprocess/synop_data/135_data.csv')
    parser.add_argument('--train_split', help='Train split factor from 0 to 1', default=0.75, type=float)
    parser.add_argument('--start_seq', help='Hour shift for start predicted sequence', default=18, type=int)
    parser.add_argument('--end_seq', help='Hour shift for end predicted sequence', default=30, type=int)

    parser.add_argument('--gfs_past', help='Number of hours before sequence which will be taken into account',
                        default=6, type=int)
    parser.add_argument('--gfs_future', help='Number of hours after sequence which will be taken into account',
                        default=6, type=int)
    parser.add_argument('--synop_length', help='Hours for Synop sequence which will be taken into account',
                        default=12, type=int)

    args = parser.parse_args()
    train_model(**vars(args))

import argparse

from preprocess.gfs_preprocess import prepare_gfs_sequence_dataset
from preprocess.synop_preprocess import prepare_synop_dataset


def prepare_data(gfs_dir: str, synop_dir: str, start_seq: int, end_seq: int, gfs_past: int, gfs_future: int,
                 synop_length: int, train_split: int):
    dataset_train = prepare_synop_dataset(synop_dir)
    gfs_dataset = prepare_gfs_sequence_dataset(gfs_dir, start_seq, end_seq, gfs_past, gfs_future)

    # least_recent_date = gfs_dataset["date"].min()
    # dataset_train = dataset_train[dataset_train["date"] >= least_recent_date - timedelta(hours=past_len)]
    # gfs_dataset = gfs_dataset[gfs_dataset["date"] > least_recent_date + timedelta(hours=future_offset)]
    # gfs_dataset = gfs_dataset.sort_values(by=["date"])
    #
    # train_synop_input, gfs_input, train_synop_label = create_sequence(dataset_train, gfs_dataset, past_len,
    #                                                                   future_offset)
    # train_size = int(train_synop_input.shape[0] * train_split_factor)
    #
    # x_train = [train_synop_input[:train_size], gfs_input[:train_size]]
    # y_train = train_synop_label[:train_size]
    # x_valid = [train_synop_input[train_size:], gfs_input[train_size:]]


def train_model():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gfs_dir', help='GFS directory', default='preprocess/wind_and_temp')
    parser.add_argument('--synop_dir', help='SYNOP directory', default='preprocess/synop_data/135_data.csv"')
    parser.add_argument('--train_split', help='Train split factor from 0 to 1', default=0.75, type=float)
    parser.add_argument('--start_seq', help='Hour shift for start predicted sequence', default=18, type=int)
    parser.add_argument('--end_seq', help='Hour shift for end predicted sequence', default=30, type=int)

    parser.add_argument('--gfs_past', help='Number of hours before sequence which will be taken into account',
                        default=6, type=int)
    parser.add_argument('--gfs_future', help='Number of hours after sequence which will be taken into account',
                        default=6, type=int)
    parser.add_argument('--synop_length', help='Hours for Synop sequence which will be taken into account',
                        default=12, type=int)

    namespace, args = parser.parse_args()
    prepare_data(**args)


if __name__ == '__main__':
    train_model()

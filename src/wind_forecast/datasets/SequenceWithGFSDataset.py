import datetime
import math
import os

import torch
import numpy as np
from tqdm import tqdm

from gfs_archive_0_25.gfs_processor.consts import FINAL_NUMPY_FILENAME_FORMAT
from gfs_archive_0_25.utils import prep_zeros_if_needed
from wind_forecast.config.register import Config
from wind_forecast.consts import DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize
from wind_forecast.util.utils import date_from_gfs_np_file, GFS_DATASET_DIR, get_point_from_GFS_slice_for_coords, \
    target_param_to_gfs_name_level


def filter_synop_data(raw_data, list_IDs):
    list_IDs = sorted(list_IDs)
    first_date = date_from_gfs_np_file(list_IDs[0])
    last_date = date_from_gfs_np_file(list_IDs[-1])

    return raw_data[(raw_data['date'] >= first_date) & (raw_data['date'] <= last_date)]


class SequenceWithGFSDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, gfs_list_IDs, train=True):
        'Initialization'
        self.list_IDs = gfs_list_IDs
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.lstm_train_parameters
        self.synop_file = config.experiment.synop_file
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.gfs_dim = config.experiment.input_size
        self.target_coords = config.experiment.target_coords

        raw_data, _, _ = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]), norm=False,
                                               dataset_dir=DATASETS_DIRECTORY)

        synop_data = filter_synop_data(raw_data, self.list_IDs)
        labels = synop_data[['date', self.target_param]].to_numpy()

        train_synop_data = synop_data[list(list(zip(*self.train_params))[1])].to_numpy()

        features = [train_synop_data[i:i + self.sequence_length, :].T for i in
                    range(train_synop_data.shape[0] - self.sequence_length - self.prediction_offset + 1)]

        targets = [labels[i + self.sequence_length + self.prediction_offset - 1] for i in
                        range(labels.shape[0] - self.sequence_length - self.prediction_offset + 1)]
        features = np.array(features).reshape((len(features), self.sequence_length, len(self.train_params)))

        self.features, self.gfs_data, self.targets = self.match_gfs_with_synop_sequence(features, targets)

        self.features, _, _ = normalize(self.features)
        self.targets, mean, std = normalize(self.targets)

        assert len(self.features) == len(self.targets)
        assert len(self.features) == len(self.gfs_data)
        length = len(self.targets)
        training_data = list(zip(zip(self.features, self.gfs_data), self.targets))[:int(length * 0.8)]
        test_data = list(zip(zip(self.features, self.gfs_data), self.targets))[int(length * 0.8):]

        if train:
            data = training_data
        else:
            data = test_data

        self.data = data

        print(mean)
        print(std)

    def match_gfs_with_synop_sequence(self, features, targets):
        gfs_values = []
        new_targets = []
        new_features = []
        for index, value in tqdm(enumerate(targets)):
            # value = [date, target_param]
            date = value[0]
            last_date_in_sequence = date - datetime.timedelta(
                hours=self.prediction_offset + 6)  # 00 run is available at 6 UTC
            day = last_date_in_sequence.day
            month = last_date_in_sequence.month
            year = last_date_in_sequence.year
            hour = int(last_date_in_sequence.hour)
            run = ['00', '06', '12', '18'][(hour // 6)]

            gfs_filename = FINAL_NUMPY_FILENAME_FORMAT.format(year, prep_zeros_if_needed(str(month), 1),
                                                              prep_zeros_if_needed(str(day), 1), run,
                                                              prep_zeros_if_needed(
                                                                  str((self.prediction_offset + 1) // 3 * 3), 2))
            if gfs_filename in self.list_IDs:  # check if there is a forecast available
                if self.target_param == 'wind_velocity':
                    val_v = get_point_from_GFS_slice_for_coords(
                        np.load(os.path.join(GFS_DATASET_DIR, 'V GRD', 'HTGL_10', gfs_filename)),
                        self.target_coords[0], self.target_coords[1])
                    val_u = get_point_from_GFS_slice_for_coords(
                        np.load(os.path.join(GFS_DATASET_DIR, 'U GRD', 'HTGL_10', gfs_filename)),
                        self.target_coords[0], self.target_coords[1])
                    val = math.sqrt(val_u ** 2 + val_v ** 2)
                else:
                    val = get_point_from_GFS_slice_for_coords(
                        np.load(os.path.join(GFS_DATASET_DIR, target_param_to_gfs_name_level(self.target_param)[0]['name'],
                                             target_param_to_gfs_name_level(self.target_param)[0]['level'], gfs_filename)),
                        self.target_coords[0], self.target_coords[1])

                gfs_values.append(val)
                new_targets.append(value[1])
                new_features.append(features[index])

        gfs_values = np.array(gfs_values)
        gfs_values = (gfs_values - np.mean(gfs_values)) / np.std(gfs_values)
        return np.array(new_features), gfs_values, np.array(new_targets)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'

        sample, label = self.data[index][0], self.data[index][1]

        x, gfs_input = sample[0], sample[1]

        return x, gfs_input, label

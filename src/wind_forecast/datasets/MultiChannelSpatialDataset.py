import os
import re
from datetime import timedelta

import torch
import numpy as np
from gfs_archive_0_25.utils import prep_zeros_if_needed
from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY, NETCDF_FILE_REGEX
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.utils import GFS_DATASET_DIR, date_from_gfs_np_file, initialize_mean_and_std, NormalizationType, \
    initialize_min_max, initialize_mean_and_std_for_sequence, initialize_min_max_for_sequence, get_values_for_sequence


class MultiChannelSpatialDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, list_IDs, train=True, normalize=True, sequence_length=1):
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.labels, self.label_mean, self.label_std = prepare_synop_dataset(self.synop_file, [self.target_param], dataset_dir=SYNOP_DATASETS_DIRECTORY)
        self.dim = config.experiment.cnn_input_size
        self.channels = len(self.train_parameters)
        self.normalization_type = config.experiment.normalization_type
        self.sequence_length = sequence_length

        if self.sequence_length > 1:
            self.initialize_list_IDs_for_sequence(list_IDs)
        else:
            self.list_IDs = list_IDs

        length = len(self.list_IDs)
        training_data, test_data = self.list_IDs[:int(length * 0.8)], self.list_IDs[int(length * 0.8):]
        if train:
            data = training_data
        else:
            data = test_data

        self.data = data
        self.mean, self.std = [], []
        self.normalize = normalize
        if normalize:
            self.normalize_data(config.experiment.normalization_type)

    def initialize_list_IDs_for_sequence(self, list_IDs: [str]):
        # filter out files, which are not continued by sufficient number of consecutive forecasts
        new_list = []
        list_IDs = sorted(list_IDs)
        param = self.train_parameters[0]
        for id in list_IDs:
            date_matcher = re.match(NETCDF_FILE_REGEX, id)
            offset = int(date_matcher.group(6))
            exists = True
            for frame in range(1, self.sequence_length):
                new_id = id.replace(f"f{prep_zeros_if_needed(str(offset), 2)}",
                                    f"f{prep_zeros_if_needed(str(offset + 3), 2)}")
                if not os.path.exists(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], new_id)):
                    exists = False
                    break
                offset = offset + 3
            if exists:
                new_list.append(id)

        self.list_IDs = new_list

    def normalize_data(self, normalization_type: NormalizationType):
        if normalization_type == NormalizationType.STANDARD:
            if self.sequence_length > 1:
                self.mean, self.std = initialize_mean_and_std_for_sequence(self.list_IDs, self.train_parameters, self.dim, self.sequence_length)
            else:
                self.mean, self.std = initialize_mean_and_std(self.list_IDs, self.train_parameters, self.dim)
        else:
            if self.sequence_length > 1:
                self.min, self.max = initialize_min_max_for_sequence(self.list_IDs, self.train_parameters, self.sequence_length)
            else:
                self.min, self.max = initialize_min_max(self.list_IDs, self.train_parameters)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.data[index]

        X, y = self.__data_generation(ID)

        return X, y

    def __data_generation(self, ID):
        # Initialization
        if self.sequence_length > 1:
            x = np.empty((self.sequence_length, self.channels, *self.dim))
            y = np.empty(self.sequence_length)

            # Generate data
            for j, param in enumerate(self.train_parameters):
                # Store sample
                x[:, j, ] = get_values_for_sequence(ID, param, self.sequence_length)
                if self.normalize:
                    if self.normalization_type == NormalizationType.STANDARD:
                        x[:, j, ] = (x[:, j, ] - self.mean[j]) / self.std[j]
                    else:
                        x[:, j,] = (x[:, j,] - self.min[j]) / (self.max[j] - self.min[j])

            first_forecast_date = date_from_gfs_np_file(ID)
            labels = [self.labels[self.labels["date"] == first_forecast_date + timedelta(hours=offset * 3)][self.target_param].values[0] for offset in range(0, self.sequence_length)]
            y[:] = labels
        else:
            x = np.empty((self.channels, *self.dim))
            y = np.empty(1)

            # Generate data
            for j, param in enumerate(self.train_parameters):
                # Store sample
                x[j, ] = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], ID))
                if self.normalize:
                    if self.normalization_type == NormalizationType.STANDARD:
                        x[j, ] = (x[j, ] - self.mean[j]) / self.std[j]
                    else:
                        x[j, ] = (x[j, ] - self.min[j]) / (self.max[j] - self.min[j])

            forecast_date = date_from_gfs_np_file(ID)
            label = self.labels[self.labels["date"] == forecast_date][self.target_param]

            y[0] = label.values[0]
        return x, y
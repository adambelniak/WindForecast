import os
from datetime import timedelta

import torch
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import GFS_DATASET_DIR, date_from_gfs_np_file, initialize_mean_and_std,\
    initialize_min_max, initialize_mean_and_std_for_sequence, initialize_min_max_for_sequence, get_GFS_values_for_sequence


class MultiChannelSpatialDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, train_IDs, labels, normalize=True):
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.labels = labels
        self.dim = config.experiment.cnn_input_size
        self.channels = len(self.train_parameters)
        self.normalization_type = config.experiment.normalization_type
        self.sequence_length = config.experiment.sequence_length
        self.list_IDs = train_IDs

        self.data = self.list_IDs
        self.mean, self.std = [], []
        self.normalize = normalize
        if normalize:
            self.normalize_data(config.experiment.normalization_type)

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
                x[:, j, ] = get_GFS_values_for_sequence(ID, param, self.sequence_length)
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
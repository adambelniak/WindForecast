import os

import torch
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.util.config import process_config
from wind_forecast.util.utils import NormalizationType, \
    initialize_mean_and_std_for_sequence, initialize_min_max_for_sequence, get_values_for_sequence, \
    initialize_GFS_list_IDs_for_sequence, CMAX_DATASET_DIR, initialize_mean_and_std_cmax, initialize_min_max_cmax


class CMAXDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, train_IDs, train=True, normalize=True):
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.dim = config.experiment.cmax_sample_size
        self.normalization_type = config.experiment.normalization_type
        self.sequence_length = config.experiment.sequence_length

        self.list_IDs = train_IDs

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

    def normalize_data(self, normalization_type: NormalizationType):
        if normalization_type == NormalizationType.STANDARD:
            # if self.sequence_length > 1:
            #     self.mean, self.std = initialize_mean_and_std_for_sequence(self.train_IDs, self.train_parameters, self.dim, self.sequence_length)
            # else:
            self.mean, self.std = initialize_mean_and_std_cmax(self.list_IDs, self.dim)
        else:
            # if self.sequence_length > 1:
            #     self.min, self.max = initialize_min_max_for_sequence(self.train_IDs, self.train_parameters, self.sequence_length)
            # else:
            self.min, self.max = initialize_min_max_cmax(self.list_IDs)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.data[index]

        X = self.__data_generation(ID)

        return X

    def __data_generation(self, ID):
        # Initialization
        # TODO reading a sequence
        # if self.sequence_length > 1:
        #     x = np.empty((self.sequence_length, self.channels, *self.dim))
        #     y = np.empty(self.sequence_length)
        #
        #     # Generate data
        #     for j, param in enumerate(self.train_parameters):
        #         # Store sample
        #         x[:, j, ] = get_values_for_sequence(ID, param, self.sequence_length)
        #         if self.normalize:
        #             if self.normalization_type == NormalizationType.STANDARD:
        #                 x[:, j, ] = (x[:, j, ] - self.mean[j]) / self.std[j]
        #             else:
        #                 x[:, j,] = (x[:, j,] - self.min[j]) / (self.max[j] - self.min[j])
        #
        #     first_forecast_date = date_from_gfs_np_file(ID)
        #     labels = [self.labels[self.labels["date"] == first_forecast_date + timedelta(hours=offset * 3)][self.target_param].values[0] for offset in range(0, self.sequence_length)]
        #     y[:] = labels
        # else:
        x = np.empty((1, *self.dim))

        # Generate data
        # TODO Load .h5 files, subtract mask, normalize
        x[0, ] = np.load(os.path.join(CMAX_DATASET_DIR, ID))
        if self.normalize:
            if self.normalization_type == NormalizationType.STANDARD:
                x[0, ] = (x[0, ] - self.mean) / self.std
            else:
                x[0, ] = (x[0, ] - self.min) / (self.max - self.min)
        return x
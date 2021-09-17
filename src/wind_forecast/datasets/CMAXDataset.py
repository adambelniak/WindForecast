import os

import torch
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.util.cmax_util import CMAX_DATASET_DIR, initialize_mean_and_std_cmax, initialize_min_max_cmax, \
    get_cmax_values_for_sequence, get_hdf
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.config import process_config


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

        self.np_mask_for_cmax = np.load(os.path.join(CMAX_DATASET_DIR, "mask.npy"))

    def normalize_data(self, normalization_type: NormalizationType):
        if normalization_type == NormalizationType.STANDARD:
            self.mean, self.std = initialize_mean_and_std_cmax(self.list_IDs, self.dim, self.sequence_length)
        else:
            self.min, self.max = initialize_min_max_cmax(self.list_IDs, self.sequence_length)

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
        if self.sequence_length > 1:
            x = np.empty((self.sequence_length, *self.dim))

            # Generate data
            x[:, ] = get_cmax_values_for_sequence(ID, self.sequence_length)
            if self.normalize:
                if self.normalization_type == NormalizationType.STANDARD:
                    x[:, ] = (x[:, ] - self.mean) / self.std
                else:
                    x[:, ] = (x[:, ] - self.min) / (self.max - self.min)

        else:
            x = np.empty((1, *self.dim))

            # Generate data
            x[0, ] = get_hdf(ID, self.np_mask_for_cmax)

            if self.normalize:
                if self.normalization_type == NormalizationType.STANDARD:
                    x[0, ] = (x[0, ] - self.mean) / self.std
                else:
                    x[0, ] = (x[0, ] - self.min) / (self.max - self.min)
        return x
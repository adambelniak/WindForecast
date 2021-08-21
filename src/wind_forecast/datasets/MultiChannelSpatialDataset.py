import os

import torch
import numpy as np

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.utils import GFS_DATASET_DIR, date_from_gfs_np_file, initialize_mean_and_std, NormalizationType, \
    initialize_min_max


class MultiChannelSpatialDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, list_IDs, train=True, normalize=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.labels, self.label_mean, self.label_std = prepare_synop_dataset(self.synop_file, [self.target_param], dataset_dir=SYNOP_DATASETS_DIRECTORY)
        self.dim = config.experiment.cnn_input_size
        self.channels = len(self.train_parameters)
        self.normalization_type = config.experiment.normalization_type

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
            if config.experiment.normalization_type == NormalizationType.STANDARD:
                self.mean, self.std = initialize_mean_and_std(self.list_IDs, self.train_parameters, self.dim)
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
                    x[j,] = (x[j,] - self.min[j]) / (self.max[j] - self.min[j])

        forecast_date = date_from_gfs_np_file(ID)
        label = self.labels[self.labels["date"] == forecast_date][self.target_param]
        if len(label) == 0:
            print(forecast_date)
        y[0] = label.values[0]
        return x, y
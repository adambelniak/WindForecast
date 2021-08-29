import os

import torch
import numpy as np

from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.utils import GFS_DATASET_DIR, date_from_gfs_np_file, initialize_mean_and_std, NormalizationType, \
    initialize_min_max, get_dim_of_GFS_slice_for_coords, get_subregion_from_GFS_slice_for_coords


class MultiChannelSpatialSubregionDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, list_IDs, train=True, normalize=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.labels, _, _ = prepare_synop_dataset(self.synop_file, [self.target_param],
                                                                             dataset_dir=SYNOP_DATASETS_DIRECTORY)
        self.subregion_coords = Coords(config.experiment.subregion_nlat,
                                       config.experiment.subregion_slat,
                                       config.experiment.subregion_wlon,
                                       config.experiment.subregion_elon)

        self.dim = get_dim_of_GFS_slice_for_coords(self.subregion_coords)

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
                self.mean, self.std = initialize_mean_and_std(self.list_IDs, self.train_parameters, self.dim, self.subregion_coords)
                print(self.mean, self.std)
            else:
                self.min, self.max = initialize_min_max(self.list_IDs, self.train_parameters, self.subregion_coords)
                print(self.min, self.max)

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
            gfs_sample = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], ID))
            # Store sample
            x[j,] = get_subregion_from_GFS_slice_for_coords(gfs_sample, self.subregion_coords)
            if self.normalize:
                if self.normalization_type == NormalizationType.STANDARD:
                    x[j,] = (x[j,] - self.mean[j]) / self.std[j]
                else:
                    x[j,] = (x[j,] - self.min[j]) / (self.max[j] - self.min[j])

        forecast_date = date_from_gfs_np_file(ID)
        label = self.labels[self.labels["date"] == forecast_date][self.target_param]
        if len(label) == 0:
            print(forecast_date)
        y[0] = label.values[0]
        return x, y

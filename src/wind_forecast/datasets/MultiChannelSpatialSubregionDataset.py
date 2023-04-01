from datetime import timedelta

import numpy as np

from util.coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import initialize_mean_and_std, initialize_min_max, \
    get_dim_of_GFS_slice_for_coords, initialize_mean_and_std_for_sequence, initialize_min_max_for_sequence, \
    get_GFS_values_for_sequence, date_from_gfs_date_key


class MultiChannelSpatialSubregionDataset(BaseDataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, train_IDs, labels, normalize=True):
        super().__init__()
        self.train_parameters = process_config(config.experiment.train_parameters_config_file).params
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.labels = labels
        self.subregion_coords = Coords(config.experiment.subregion_nlat,
                                       config.experiment.subregion_slat,
                                       config.experiment.subregion_wlon,
                                       config.experiment.subregion_elon)

        self.prediction_offset = config.experiment.prediction_offset
        self.dim = get_dim_of_GFS_slice_for_coords(self.subregion_coords)

        self.channels = len(self.train_parameters)
        self.normalization_type = config.experiment.normalization_type
        self.sequence_length = config.experiment.sequence_length

        self.list_IDs = train_IDs

        self.data = self.list_IDs[str(self.prediction_offset)]
        self.normalize = normalize
        if normalize:
            self.normalize_data(config.experiment.normalization_type)

    def normalize_data(self, normalization_type: NormalizationType):
        if normalization_type == NormalizationType.STANDARD:
            if self.sequence_length > 1:
                self.mean, self.std = initialize_mean_and_std_for_sequence(self.list_IDs, self.train_parameters,
                                                                           self.dim, self.sequence_length,
                                                                           self.prediction_offset, self.subregion_coords)
            else:
                self.mean, self.std = initialize_mean_and_std(self.list_IDs, self.train_parameters, self.dim, self.prediction_offset,
                                                              self.subregion_coords)
        else:
            if self.sequence_length > 1:
                self.min, self.max = initialize_min_max_for_sequence(self.list_IDs, self.train_parameters,
                                                                     self.sequence_length, self.prediction_offset, self.subregion_coords)
            else:
                self.min, self.max = initialize_min_max(self.list_IDs, self.train_parameters, self.prediction_offset, self.subregion_coords)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.data[index]

        X, y = self.__data_generation(ID)

        return X, np.expand_dims(np.array(y[-1]), axis=0)

    def __data_generation(self, ID):
        # Initialization
        if self.sequence_length > 1:
            x = np.empty((self.sequence_length, self.channels, *self.dim))
            y = np.empty(self.sequence_length)

            # Generate data
            for j, param in enumerate(self.train_parameters):
                # Store sample
                x[:, j, ] = get_GFS_values_for_sequence(ID, param, self.sequence_length, self.prediction_offset, self.subregion_coords)
                if self.normalize:
                    if self.normalization_type == NormalizationType.STANDARD:
                        x[:, j, ] = (x[:, j, ] - self.mean[j]) / self.std[j]
                    else:
                        x[:, j, ] = (x[:, j, ] - self.min[j]) / (self.max[j] - self.min[j])

            first_forecast_date = date_from_gfs_date_key(ID)
            labels = [self.labels[self.labels["date"] == first_forecast_date + timedelta(hours=offset * 3)][
                          self.target_param].values[0] for offset in range(0, self.sequence_length)]
            y[:] = labels
        else:
            x = np.empty((self.channels, *self.dim))
            y = np.empty(1)

            # Generate data
            for j, param in enumerate(self.train_parameters):
                # Store sample
                x[j,] = get_GFS_values_for_sequence(ID, param, self.sequence_length, self.prediction_offset, self.subregion_coords)
                if self.normalize:
                    if self.normalization_type == NormalizationType.STANDARD:
                        x[j,] = (x[j,] - self.mean[j]) / self.std[j]
                    else:
                        x[j,] = (x[j,] - self.min[j]) / (self.max[j] - self.min[j])

            forecast_date = date_from_gfs_date_key(ID)
            label = self.labels[self.labels["date"] == forecast_date][self.target_param]

            y[0] = label.values[0]
        return x, y

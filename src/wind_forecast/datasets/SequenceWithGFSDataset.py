import math

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from wind_forecast.config.register import Config
from wind_forecast.preprocess.synop.synop_preprocess import normalize_synop_data
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.gfs_util import add_param_to_train_params, match_gfs_with_synop_sequence, \
    target_param_to_gfs_name_level


class SequenceWithGFSDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, config: Config, synop_data, dates, train=True, normalize_synop=True):
        """Initialization"""
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.synop_file = config.experiment.synop_file
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.target_coords = config.experiment.target_coords
        self.synop_data = synop_data.reset_index()
        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index

        all_params = add_param_to_train_params(self.train_params, self.target_param)
        feature_names = list(list(zip(*all_params))[1])

        if normalize_synop:
            # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
            self.synop_data, synop_feature_1, synop_feature_2 = normalize_synop_data(self.synop_data,
                                                                                     synop_data_indices,
                                                                                     feature_names,
                                                                                     self.sequence_length + self.prediction_offset,
                                                                                     config.experiment.normalization_type)
            target_param_index = [x for x in feature_names].index(self.target_param)
            print(synop_feature_1[target_param_index])
            print(synop_feature_2[target_param_index])

        synop_data_dates = self.synop_data['date']
        # labels and dates - dates are needed for matching the labels against GFS dates
        labels = pd.concat([synop_data_dates, self.synop_data[self.target_param]], axis=1)

        self.features = []
        targets = []
        train_params = list(list(zip(*self.train_params))[1])
        for index in tqdm(synop_data_indices):
            self.features.append(self.synop_data.iloc[index:index + self.sequence_length][train_params].to_numpy())
            targets.append(labels.iloc[index + self.sequence_length + self.prediction_offset])

        self.features, self.gfs_data, self.targets = match_gfs_with_synop_sequence(self.features, targets,
                                                                                   self.target_coords[0],
                                                                                   self.target_coords[1],
                                                                                   self.prediction_offset,
                                                                                   target_param_to_gfs_name_level(
                                                                                       self.target_param))

        if self.target_param == "wind_velocity":
            self.gfs_data = np.array([math.sqrt(velocity[0] ** 2 + velocity[1] ** 2) for velocity in self.gfs_data])
        else:
            self.gfs_data = np.array([value[0] for value in self.gfs_data])

        if config.experiment.normalization_type == NormalizationType.STANDARD:
            self.gfs_data = (self.gfs_data - np.mean(self.gfs_data)) / np.std(self.gfs_data)
        else:
            self.gfs_data = (self.gfs_data - np.min(self.gfs_data)) / (np.max(self.gfs_data) - np.min(self.gfs_data))

        assert len(self.features) == len(self.targets)
        assert len(self.features) == len(self.gfs_data)
        length = len(self.targets)
        training_data = list(zip(zip(self.features, self.gfs_data), self.targets))[:int(length * 0.8)]
        # do not use any frame from train set in test set
        test_data = list(zip(zip(self.features, self.gfs_data), self.targets))[int(length * 0.8) + self.sequence_length - 1:]

        if train:
            data = training_data
        else:
            data = test_data

        self.data = data

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""

        sample, label = self.data[index][0], self.data[index][1]

        x, gfs_input = sample[0], sample[1]

        return x, gfs_input, label

import torch
from tqdm import tqdm
import numpy as np
import math

from wind_forecast.config.register import Config
from wind_forecast.preprocess.synop.synop_preprocess import normalize_synop_data
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.gfs_util import add_param_to_train_params, \
    target_param_to_gfs_name_level, match_gfs_with_synop_sequence2sequence
import pandas as pd


class Sequence2SequenceWithGFSDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, synop_data, dates, train=True, normalize_synop=True):
        'Initialization'
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.target_coords = config.experiment.target_coords
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.mean = ...
        self.std = ...
        synop_data = synop_data.reset_index()

        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        synop_data_indices = synop_data[synop_data["date"].isin(dates)].index
        params = add_param_to_train_params(self.train_params, self.target_param)
        feature_names = list(list(zip(*params))[1])
        target_param_index = [x for x in feature_names].index(self.target_param)

        if normalize_synop:
            # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
            synop_data, synop_mean, synop_std = normalize_synop_data(synop_data, synop_data_indices,
                                                                          feature_names,
                                                                          self.sequence_length + self.prediction_offset
                                                                          + self.future_sequence_length,
                                                                          config.experiment.normalization_type)
            self.mean = synop_mean[target_param_index]
            self.std = synop_std[target_param_index]
            print(synop_mean[target_param_index])
            print(synop_std[target_param_index])

        synop_data_dates = synop_data['date']
        # all_targets and dates - dates are needed for matching the labels against GFS dates

        self.features = []
        all_targets = []
        train_params = list(list(zip(*self.train_params))[1])
        all_targets_and_labels = pd.concat([synop_data_dates, synop_data[train_params]], axis=1)
        for index in tqdm(synop_data_indices):
            self.features.append(synop_data.iloc[index:index + self.sequence_length][train_params].to_numpy())
            all_targets.append(all_targets_and_labels.iloc[
                               index + self.sequence_length + self.prediction_offset:index + self.sequence_length + self.prediction_offset + self.future_sequence_length])

        self.features, self.gfs_data, self.all_targets = match_gfs_with_synop_sequence2sequence(self.features, all_targets,
                                                                                       self.target_coords[0],
                                                                                       self.target_coords[1],
                                                                                       self.prediction_offset,
                                                                                       target_param_to_gfs_name_level(
                                                                                           self.target_param))

        self.targets = [target[:, target_param_index] for target in self.all_targets]

        if self.target_param == "wind_velocity":
            self.gfs_data = np.array([math.sqrt(velocity[0] ** 2 + velocity[1] ** 2) for velocity in self.gfs_data])

        if config.experiment.normalization_type == NormalizationType.STANDARD:
            self.gfs_data = (self.gfs_data - np.mean(self.gfs_data)) / np.std(self.gfs_data)
        else:
            self.gfs_data = (self.gfs_data - np.min(self.gfs_data)) / (np.max(self.gfs_data) - np.min(self.gfs_data))

        assert len(self.features) == len(self.targets)
        assert len(self.features) == len(self.all_targets)
        assert len(self.features) == len(self.gfs_data)
        length = len(self.targets)
        print(length)

        training_data = list(zip(zip(self.features, self.gfs_data, self.all_targets), self.targets))[:int(length * 0.8)]
        # do not use any frame from train set in test set
        test_data = list(zip(zip(self.features, self.gfs_data, self.all_targets), self.targets))[int(length * 0.8) + self.sequence_length - 1:]

        if train:
            self.data = training_data
        else:
            self.data = test_data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        x, label = self.data[index][0], self.data[index][1]
        inputs, gfs_inputs, all_targets = x[0], x[1], x[2]
        return inputs, gfs_inputs, all_targets, label

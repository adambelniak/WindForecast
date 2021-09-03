import torch
import numpy as np
import pandas as pd

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize
from wind_forecast.util.utils import add_param_to_train_params, match_gfs_with_synop_sequence, \
    target_param_to_gfs_name_level


class SequenceLimitedToGFSDatesDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, config: Config, train=True):
        """Initialization"""
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.synop_file = config.experiment.synop_file
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.target_coords = config.experiment.target_coords

        params = add_param_to_train_params(self.train_params, self.target_param)
        feature_names = list(list(zip(*params))[1])

        synop_data, _, _ = prepare_synop_dataset(self.synop_file, feature_names, norm=False,
                                                 dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=2015)

        synop_data_dates = synop_data['date']
        # normalize here to keep the param_name <> value mapping
        synop_data, synop_feature_1, synop_feature_2 = normalize(synop_data[feature_names], config.experiment.normalization_type)

        labels = pd.concat([synop_data_dates, synop_data[self.target_param]], axis=1).to_numpy()

        train_synop_data = synop_data[list(list(zip(*self.train_params))[1])].to_numpy()

        features = [train_synop_data[i:i + self.sequence_length, :].T for i in
                    range(train_synop_data.shape[0] - self.sequence_length - self.prediction_offset + 1)]

        targets = [labels[i + self.sequence_length + self.prediction_offset - 1] for i in
                   range(labels.shape[0] - self.sequence_length - self.prediction_offset + 1)]
        features = np.transpose(np.array(features), (0, 2, 1))

        self.features, self.targets = match_gfs_with_synop_sequence(features, targets,
                                                                                   self.target_coords[0],
                                                                                   self.target_coords[1],
                                                                                   self.prediction_offset,
                                                                                   target_param_to_gfs_name_level(self.target_param), return_GFS=False)

        length = len(self.targets)
        training_data = list(zip(self.features, self.targets))[:int(length * 0.8)]
        test_data = list(zip(self.features, self.targets))[int(length * 0.8):]

        if train:
            data = training_data
        else:
            data = test_data

        self.data = data

        print(synop_feature_1)
        print(synop_feature_2)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""

        x, label = self.data[index][0], self.data[index][1]

        return x, label

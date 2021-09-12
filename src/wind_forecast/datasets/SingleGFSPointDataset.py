import pandas as pd
import torch
import numpy as np

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.utils import match_gfs_with_synop_sequence, NormalizationType


class SingleGFSPointDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, config: Config, train=True):
        """Initialization"""
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.prediction_offset = config.experiment.prediction_offset
        self.target_coords = config.experiment.target_coords

        synop_data, synop_mean, synop_std = prepare_synop_dataset(self.synop_file, [self.target_param],
                                                                  dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                                  from_year=config.experiment.synop_from_year)

        synop_data_dates = synop_data['date']
        labels = pd.concat([synop_data_dates, synop_data[self.target_param]], axis=1).to_numpy().tolist()
        _, self.gfs_data, self.targets = match_gfs_with_synop_sequence(labels, labels,
                                                                       self.target_coords[0],
                                                                       self.target_coords[1],
                                                                       self.prediction_offset,
                                                                       self.train_parameters,
                                                                       exact_date_match=True)

        self.targets = self.targets.reshape((len(self.targets), 1))

        if config.experiment.normalization_type == NormalizationType.STANDARD:
            self.gfs_data = (self.gfs_data - np.mean(self.gfs_data, axis=0)) / np.std(self.gfs_data, axis=0)
        else:
            self.gfs_data = (self.gfs_data - np.min(self.gfs_data, axis=0)) / (np.max(self.gfs_data, axis=0) - np.min(self.gfs_data, axis=0))

        assert len(self.gfs_data) == len(self.targets)
        length = len(self.targets)
        training_data = list(zip(self.gfs_data, self.targets))[:int(length * 0.8)]
        test_data = list(zip(self.gfs_data, self.targets))[int(length * 0.8):]

        if train:
            data = training_data
        else:
            data = test_data

        self.data = data
        print(synop_mean)
        print(synop_std)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""

        x, label = self.data[index][0], self.data[index][1]

        return x, label

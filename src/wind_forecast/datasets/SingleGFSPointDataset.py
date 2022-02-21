import pandas as pd
import torch
import numpy as np

from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import GFSUtil, target_param_to_gfs_name_level


class SingleGFSPointDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, config: Config):
        """Initialization"""
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.target_param = config.experiment.target_parameter
        self.synop_file = config.experiment.synop_file
        self.prediction_offset = config.experiment.prediction_offset
        self.sequence_length = config.experiment.sequence_length
        coords = config.experiment.target_coords
        self.target_coords = Coords(coords[0], coords[0], coords[1], coords[1])
        self.gfs_util = GFSUtil(self.target_coords, self.sequence_length, 0, self.prediction_offset, self.train_parameters,
                                target_param_to_gfs_name_level(self.target_param))

        synop_data, synop_mean, synop_std = prepare_synop_dataset(self.synop_file, [self.target_param],
                                                                  dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                                  from_year=config.experiment.synop_from_year,
                                                                  to_year=config.experiment.synop_to_year)

        synop_data_dates = synop_data['date']
        labels = pd.concat([synop_data_dates, synop_data[self.target_param]], axis=1).to_numpy().tolist()
        _, self.gfs_data, self.targets = self.gfs_util.match_gfs_with_synop_sequence(labels, labels)

        self.targets = self.targets.reshape((len(self.targets), 1))

        if config.experiment.normalization_type == NormalizationType.STANDARD:
            self.gfs_data = (self.gfs_data - np.mean(self.gfs_data, axis=0)) / np.std(self.gfs_data, axis=0)
        else:
            self.gfs_data = (self.gfs_data - np.min(self.gfs_data, axis=0)) / (np.max(self.gfs_data, axis=0) - np.min(self.gfs_data, axis=0))

        assert len(self.gfs_data) == len(self.targets)
        self.data = list(zip(self.gfs_data, self.targets))
        print(synop_mean)
        print(synop_std)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""

        x, label = self.data[index][0], self.data[index][1]

        return x, label

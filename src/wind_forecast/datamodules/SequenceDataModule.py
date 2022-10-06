import math
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.datasets.SequenceWithGFSDataset import SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize_synop_data_for_training
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import add_param_to_train_params, normalize_gfs_data, \
    target_param_to_gfs_name_level, GFSUtil
from wind_forecast.util.logging import log
from wind_forecast.util.synop_util import get_correct_dates_for_sequence


class SequenceDataModule(SplittableDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)
        self.config = config
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle
        self.synop_file = config.experiment.synop_file
        self.train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        all_params = add_param_to_train_params(self.train_params, self.target_param)
        self.feature_names = list(list(zip(*all_params))[1])
        self.target_param_index = [x for x in self.feature_names].index(self.target_param)
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.normalization_type = config.experiment.normalization_type
        self.target_coords = config.experiment.target_coords
        self.synop_from_year = config.experiment.synop_from_year
        self.synop_to_year = config.experiment.synop_to_year
        self.periodic_features = config.experiment.synop_periodic_features
        self.gfs_train_params = process_config(
            config.experiment.train_parameters_config_file).params if config.experiment.use_gfs_data else None
        self.gfs_target_params = self.gfs_train_params if config.experiment.use_gfs_data else target_param_to_gfs_name_level(
            self.target_param)

        coords = config.experiment.target_coords
        self.target_coords = Coords(coords[0], coords[0], coords[1], coords[1])
        self.gfs_util = GFSUtil(self.target_coords, self.sequence_length, 0, self.prediction_offset,
                                self.gfs_train_params,
                                self.gfs_target_params)
        self.synop_data = ...
        self.synop_data_indices = ...
        self.removed_dataset_indices = []
        self.synop_mean = ...
        self.synop_std = ...

    def prepare_data(self, *args, **kwargs):
        self.synop_data = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]),
                                                dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                from_year=self.synop_from_year,
                                                to_year=self.synop_to_year,
                                                norm=False)

        dates = get_correct_dates_for_sequence(self.synop_data, self.sequence_length, 1, self.prediction_offset)

        self.synop_data = self.synop_data.reset_index()
        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index
        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, self.synop_feature_names, synop_mean, synop_std = normalize_synop_data_for_training(
            self.synop_data, self.synop_data_indices,
            self.feature_names,
            self.sequence_length + self.prediction_offset,
            self.normalization_type,
            self.periodic_features)
        log.info(f"Synop mean: {synop_mean[self.target_param]}")
        log.info(f"Synop std: {synop_std[self.target_param]}")

    def setup(self, stage: Optional[str] = None):
        if self.get_from_cache(stage):
            return

        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, synop_targets = self.prepare_dataset_for_gfs()
            if self.gfs_train_params is not None:
                dataset = SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets, all_gfs_input_data)
            else:
                dataset = SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets)
        else:
            dataset = SequenceDataset(config=self.config, synop_data=self.synop_data,
                                      synop_data_indices=self.synop_data_indices)

        self.split_dataset(self.config, dataset, self.sequence_length)

    def prepare_dataset_for_gfs(self):
        log.info("Preparing the dataset")
        synop_inputs, all_synop_targets = self.resolve_all_synop_data()

        all_gfs_input_data = ...
        if self.gfs_train_params is not None:
            # first, get GFS input data that matches synop input data
            synop_inputs, all_gfs_input_data, _, removed_indices = self.gfs_util.match_gfs_with_synop_sequence(
                synop_inputs,
                synop_inputs)

        # Then, get GFS data for forecast frames
        synop_inputs, gfs_target_data, all_synop_targets, next_removed_indices = self.gfs_util.match_gfs_with_synop_sequence(
            synop_inputs, all_synop_targets)

        self.removed_dataset_indices.extend(next_removed_indices)

        if self.gfs_train_params is not None:
            all_gfs_input_data = [item for index, item in enumerate(all_gfs_input_data) if
                                  index not in next_removed_indices]

        synop_targets = [target[:, self.target_param_index] for target in all_synop_targets]

        if self.target_param == "wind_velocity":
            gfs_target_data = np.array([math.sqrt(velocity[0] ** 2 + velocity[1] ** 2) for velocity in gfs_target_data])

        gfs_target_data = normalize_gfs_data(gfs_target_data, self.normalization_type)

        assert len(synop_inputs) == len(synop_targets)
        assert len(synop_inputs) == len(all_synop_targets)
        assert len(synop_inputs) == len(gfs_target_data)

        if self.gfs_train_params is not None:
            assert len(synop_inputs) == len(all_gfs_input_data)
            return synop_inputs, all_gfs_input_data, gfs_target_data, synop_targets

        return synop_inputs, [], gfs_target_data, synop_targets

    def resolve_all_synop_data(self):
        synop_inputs = []
        all_synop_targets = []
        synop_data_dates = self.synop_data['date']
        train_params = list(list(zip(*self.train_params))[1])
        # all_targets and dates - dates are needed for matching the labels against GFS dates
        all_targets_and_labels = pd.concat([synop_data_dates, self.synop_data[train_params]], axis=1)

        for index in tqdm(self.synop_data_indices):
            synop_inputs.append(
                self.synop_data.iloc[index:index + self.sequence_length][train_params].to_numpy())
            all_synop_targets.append(all_targets_and_labels.iloc[index + self.sequence_length + self.prediction_offset])

        return synop_inputs, all_synop_targets

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

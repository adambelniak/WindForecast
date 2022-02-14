import math
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.datasets.SequenceWithGFSDataset import SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize_synop_data_for_training
from wind_forecast.util.common_util import split_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import add_param_to_train_params, normalize_gfs_data, match_gfs_with_synop_sequence, \
    target_param_to_gfs_name_level
from wind_forecast.util.synop_util import get_correct_dates_for_sequence
import pandas as pd
import numpy as np


class SequenceDataModule(LightningDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__()
        self.config = config
        self.val_split = config.experiment.val_split
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
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
        self.gfs_train_params = process_config(
            config.experiment.train_parameters_config_file) if config.experiment.use_all_gfs_params else None

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
        self.synop_data, self.synop_feature_names, synop_mean, synop_std = normalize_synop_data_for_training(self.synop_data, self.synop_data_indices,
                                                                                                             self.feature_names,
                                                                                                             self.sequence_length + self.prediction_offset,
                                                                                                             self.normalization_type)
        print(f"Synop mean: {synop_mean[self.target_param_index]}")
        print(f"Synop std: {synop_std[self.target_param_index]}")

    def setup(self, stage: Optional[str] = None):
        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, synop_targets = self.prepare_dataset_for_gfs()
            if self.gfs_train_params is not None:
                dataset = SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets, all_gfs_input_data)
            else:
                dataset = SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets)
        else:
            dataset = SequenceDataset(config=self.config, synop_data=self.synop_data, synop_data_indices=self.synop_data_indices)

        self.dataset_train, self.dataset_val = split_dataset(dataset, self.config.experiment.val_split,
                                                             sequence_length=self.sequence_length if self.sequence_length > 1 else None)
        self.dataset_test = self.dataset_val

    def prepare_dataset_for_gfs(self):
        print("Preparing the dataset")
        synop_inputs, all_synop_targets = self.resolve_all_synop_data()

        all_gfs_input_data = ...
        if self.gfs_train_params is not None:
            # first, get GFS input data that matches synop input data
            synop_inputs, all_gfs_input_data, _, removed_indices = match_gfs_with_synop_sequence(synop_inputs,
                                                                      synop_inputs, self.target_coords[0],
                                                                      self.target_coords[1], 0, self.gfs_train_params)

        # Then, get GFS data for forecast frames
        synop_inputs, gfs_target_data, all_synop_targets, next_removed_indices = match_gfs_with_synop_sequence(
                                                                                    synop_inputs, all_synop_targets,
                                                                                    self.target_coords[0],
                                                                                    self.target_coords[1],
                                                                                    self.prediction_offset,
                                                                                    target_param_to_gfs_name_level(
                                                                                        self.target_param))

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

import math
from typing import Optional, Tuple, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.DataModulesCache import DataModulesCache
from wind_forecast.datasets.Sequence2SequenceDataset import Sequence2SequenceDataset
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize_synop_data
from wind_forecast.util.common_util import split_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import add_param_to_train_params, match_gfs_with_synop_sequence2sequence, \
    target_param_to_gfs_name_level, normalize_gfs_data
from wind_forecast.util.synop_util import get_correct_dates_for_sequence
import pandas as pd
import numpy as np
from wind_forecast.consts import BatchKeys


class Sequence2SequenceDataModule(LightningDataModule):

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

        self.train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        all_params = add_param_to_train_params(self.train_params, self.target_param)
        self.feature_names = list(list(zip(*all_params))[1])
        self.target_param_index = [x for x in self.feature_names].index(self.target_param)
        self.removed_dataset_indices = []

        self.synop_file = config.experiment.synop_file
        self.synop_from_year = config.experiment.synop_from_year
        self.synop_to_year = config.experiment.synop_to_year
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.normalization_type = config.experiment.normalization_type
        self.prediction_offset = config.experiment.prediction_offset
        self.target_coords = config.experiment.target_coords
        self.use_all_gfs_params = config.experiment.use_all_gfs_params
        self.gfs_train_params = process_config(
            config.experiment.train_parameters_config_file) if self.use_all_gfs_params else None
        self.gfs_target_param_indices = [self.gfs_train_params.index(param) for param in target_param_to_gfs_name_level(
            self.target_param)] if self.use_all_gfs_params else None
        self.periodic_features = config.experiment.periodic_features

        self.synop_data = ...
        self.synop_data_indices = ...
        self.synop_mean = ...
        self.synop_std = ...

    def prepare_data(self, *args, **kwargs):
        self.synop_data = prepare_synop_dataset(self.synop_file,
                                                list(list(zip(*self.train_params))[1]),
                                                dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                from_year=self.synop_from_year,
                                                to_year=self.synop_to_year,
                                                norm=False)

        if self.config.debug_mode:
            self.synop_data = self.synop_data.head(100)

        dates = get_correct_dates_for_sequence(self.synop_data, self.sequence_length, self.future_sequence_length,
                                               self.prediction_offset)

        self.synop_data = self.synop_data.reset_index()

        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index
        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, target_param_mean, target_param_std = normalize_synop_data(self.synop_data, self.synop_data_indices,
                                                                      self.feature_names,
                                                                      self.sequence_length + self.prediction_offset
                                                                      + self.future_sequence_length,
                                                                      self.target_param, self.normalization_type, self.periodic_features)
        self.synop_mean = target_param_mean
        self.synop_std = target_param_std
        print(f"Synop mean: {target_param_mean}")
        print(f"Synop std: {target_param_std}")

    def setup(self, stage: Optional[str] = None):
        cached_dataset = DataModulesCache().get_cached_dataset()
        if stage == 'test' and cached_dataset is not None:
            self.dataset_test = cached_dataset
            return

        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, all_gfs_target_data = self.prepare_dataset_for_gfs()

            if self.use_all_gfs_params:
                dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.synop_data_indices,
                                                          gfs_target_data, all_gfs_target_data, all_gfs_input_data)
            else:
                dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.synop_data_indices,
                                                          gfs_target_data)

        else:
            dataset = Sequence2SequenceDataset(self.config, self.synop_data, self.synop_data_indices)

        if len(dataset) == 0:
            raise RuntimeError("There are no valid samples in the dataset! Please check your run configuration")

        dataset.set_mean(self.synop_mean)
        dataset.set_std(self.synop_std)
        self.dataset_train, self.dataset_val = split_dataset(dataset, self.config.experiment.val_split,
                                                             sequence_length=self.sequence_length if self.sequence_length > 1 else None)
        self.dataset_test = self.dataset_val
        DataModulesCache().cache_dataset(self.dataset_test)

    def prepare_dataset_for_gfs(self):
        print("Preparing the dataset")
        # match GFS and synop sequences
        self.synop_data_indices, self.removed_dataset_indices, all_gfs_input_data, all_gfs_target_data = match_gfs_with_synop_sequence2sequence(
            self.synop_data,
            self.synop_data_indices,
            self.target_coords,
            self.sequence_length,
            self.prediction_offset,
            self.future_sequence_length,
            self.gfs_train_params,
            self.gfs_train_params if self.use_all_gfs_params
            else target_param_to_gfs_name_level(self.target_param))

        if self.use_all_gfs_params:
            # normalize GFS parameters data
            all_gfs_input_data = normalize_gfs_data(all_gfs_input_data, self.normalization_type, (0, 1))
            gfs_target_data = all_gfs_target_data[:, :, self.gfs_target_param_indices]
            all_gfs_target_data = (all_gfs_target_data - np.mean(all_gfs_target_data, axis=(0, 1))) / np.std(
                all_gfs_target_data, axis=(0, 1))
        else:
            gfs_target_data = all_gfs_target_data

        if self.target_param == "wind_velocity":
            # handle target wind_velocity forecasted by GFS
            gfs_target_data = np.apply_along_axis(lambda velocity: [math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)], -1, gfs_target_data)

        gfs_target_data = (gfs_target_data - np.mean(gfs_target_data, axis=(0, 1))) / np.std(gfs_target_data, axis=(0, 1))

        assert len(self.synop_data_indices) == len(all_gfs_target_data), f"len(all_gfs_target_data) should be {len(self.synop_data_indices)} but was {len(all_gfs_target_data)}"

        if self.use_all_gfs_params:
            assert len(self.synop_data_indices) == len(all_gfs_input_data), f"len(all_gfs_input_data) should be {len(self.synop_data_indices)} but was {len(all_gfs_input_data)}"
            return self.synop_data_indices, all_gfs_input_data, gfs_target_data, all_gfs_target_data

        return self.synop_data_indices, None, gfs_target_data, None

    def resolve_all_synop_data(self):
        synop_inputs = []
        all_synop_targets = []
        synop_data_dates = self.synop_data['date']
        train_params = list(list(zip(*self.train_params))[1])
        # all_targets and dates - dates are needed for matching the labels against GFS dates
        all_targets_and_labels = pd.concat([synop_data_dates, self.synop_data[train_params]], axis=1)

        for index in tqdm(self.synop_data_indices):
            synop_inputs.append(
                self.synop_data.iloc[index:index + self.sequence_length][[*train_params, 'date']])
            all_synop_targets.append(all_targets_and_labels.iloc[
                                     index + self.sequence_length + self.prediction_offset:index + self.sequence_length + self.prediction_offset + self.future_sequence_length])

        return synop_inputs, all_synop_targets

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, x: List[Tuple]):
        tensors, dates = [item[:-2] for item in x], [item[-2:] for item in x]
        all_data = [*default_collate(tensors), *list(zip(*dates))]
        dict_data = {
            BatchKeys.SYNOP_INPUTS.value: all_data[0],
            BatchKeys.SYNOP_TARGETS.value: all_data[1],
            BatchKeys.ALL_SYNOP_TARGETS.value: all_data[2]
        }

        if self.config.experiment.use_gfs_data:
            if self.use_all_gfs_params:
                dict_data[BatchKeys.GFS_INPUTS.value] = all_data[3]
                dict_data[BatchKeys.GFS_TARGETS.value] = all_data[4]
                dict_data[BatchKeys.ALL_GFS_TARGETS.value] = all_data[5]
                dict_data[BatchKeys.DATES_INPUTS.value] = all_data[6]
                dict_data[BatchKeys.DATES_TARGETS.value] = all_data[7]

            else:
                dict_data[BatchKeys.GFS_TARGETS.value] = all_data[3]
                dict_data[BatchKeys.DATES_INPUTS.value] = all_data[4]
                dict_data[BatchKeys.DATES_TARGETS.value] = all_data[5]

        else:
            dict_data[BatchKeys.DATES_INPUTS.value] = all_data[3]
            dict_data[BatchKeys.DATES_TARGETS.value] = all_data[4]
        return dict_data

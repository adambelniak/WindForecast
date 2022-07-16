import math
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.datasets.Sequence2SequenceDataset import Sequence2SequenceDataset
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize_synop_data_for_training
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import add_param_to_train_params, target_param_to_gfs_name_level, normalize_gfs_data, \
    GFSUtil, extend_wind_components
from wind_forecast.util.logging import log
from wind_forecast.util.synop_util import get_correct_dates_for_sequence


class Sequence2SequenceDataModule(SplittableDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)
        self.config = config
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle

        self.train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        all_params = add_param_to_train_params(self.train_params, self.target_param)
        self.feature_names = list(list(zip(*all_params))[1])

        self.synop_file = config.experiment.synop_file
        self.synop_from_year = config.experiment.synop_from_year
        self.synop_to_year = config.experiment.synop_to_year
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.normalization_type = config.experiment.normalization_type
        self.prediction_offset = config.experiment.prediction_offset
        coords = config.experiment.target_coords
        self.target_coords = Coords(coords[0], coords[0], coords[1], coords[1])

        self.gfs_train_params = process_config(config.experiment.train_parameters_config_file)
        self.gfs_target_params = self.gfs_train_params

        self.gfs_target_param_indices = [self.gfs_train_params.index(param) for param in target_param_to_gfs_name_level(
            self.target_param)]

        self.gfs_wind_components_indices = [self.gfs_train_params.index(param) for param in target_param_to_gfs_name_level(
            'wind_direction')]
        self.gfs_util = GFSUtil(self.target_coords, self.sequence_length, self.future_sequence_length,
                                self.prediction_offset, self.gfs_train_params, self.gfs_target_params)

        self.periodic_features = config.experiment.synop_periodic_features
        self.uses_future_sequences = True

        self.synop_data = ...
        self.synop_data_indices = ...
        self.synop_mean = ...
        self.synop_std = ...
        self.synop_feature_names = ...

    def prepare_data(self, *args, **kwargs):
        self.load_from_disk(self.config)

        if self.initialized:
            return

        self.synop_data = prepare_synop_dataset(self.synop_file,
                                                list(list(zip(*self.train_params))[1]),
                                                dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                from_year=self.synop_from_year,
                                                to_year=self.synop_to_year,
                                                norm=False)

        if self.config.debug_mode or self.config.tune_mode:
            self.synop_data = self.synop_data.head(self.sequence_length * 20)

        dates = get_correct_dates_for_sequence(self.synop_data, self.sequence_length, self.future_sequence_length,
                                               self.prediction_offset)

        self.synop_data = self.synop_data.reset_index()

        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index
        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, self.synop_feature_names, mean, std = normalize_synop_data_for_training(
            self.synop_data, self.synop_data_indices,
            self.feature_names,
            self.sequence_length + self.prediction_offset + self.future_sequence_length,
            self.normalization_type, self.periodic_features)
        self.synop_mean = mean[self.target_param]
        self.synop_std = std[self.target_param]
        log.info(f"Synop mean: {self.synop_mean}")
        log.info(f"Synop std: {self.synop_std}")

    def setup(self, stage: Optional[str] = None):
        if self.initialized:
            return
        if self.get_from_cache(stage):
            return

        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, all_gfs_target_data = self.prepare_dataset_for_gfs()

            dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.synop_data_indices,
                                                      self.synop_feature_names, gfs_target_data,
                                                      all_gfs_target_data, all_gfs_input_data)

        else:
            dataset = Sequence2SequenceDataset(self.config, self.synop_data, self.synop_data_indices,
                                               self.synop_feature_names)

        if len(dataset) == 0:
            raise RuntimeError("There are no valid samples in the dataset! Please check your run configuration")

        dataset.set_mean(self.synop_mean)
        dataset.set_std(self.synop_std)
        self.split_dataset(self.config, dataset, self.sequence_length)

    def prepare_dataset_for_gfs(self):
        log.info("Preparing the dataset")
        # match GFS and synop sequences
        self.synop_data_indices, all_gfs_input_data, all_gfs_target_data = self.gfs_util.match_gfs_with_synop_sequence2sequence(
            self.synop_data,
            self.synop_data_indices)

        # save target data
        gfs_target_data = all_gfs_target_data[:, :, self.gfs_target_param_indices]

        # normalize GFS parameters data
        param_names = [x['name'] for x in self.gfs_train_params]
        if "V GRD" in param_names and "U GRD" in param_names:
            all_gfs_input_data = self.prepare_gfs_data_with_wind_components(all_gfs_input_data)
            all_gfs_target_data = self.prepare_gfs_data_with_wind_components(all_gfs_target_data)
        else:
            all_gfs_input_data = normalize_gfs_data(all_gfs_input_data, self.normalization_type, (0, 1))
            all_gfs_target_data = normalize_gfs_data(all_gfs_target_data, self.normalization_type, (0, 1))

        if self.target_param == "wind_velocity":
            # handle target wind_velocity forecast by GFS
            # velocity[0] is V GRD (northward), velocity[1] is U GRD (eastward)
            gfs_target_data = np.apply_along_axis(lambda velocity: [math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)], -1,
                                                  gfs_target_data)
        if self.target_param == "wind_direction":
            # set sin and cos components as targets, do not normalize them
            gfs_target_data = np.apply_along_axis(lambda velocity: [-velocity[1] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2)),
                                                                    -velocity[0] / (math.sqrt(velocity[0] ** 2 + velocity[1] ** 2))], -1,
                                                  gfs_target_data)

        if self.target_param != "wind_direction":
            gfs_target_data = (gfs_target_data - np.mean(gfs_target_data, axis=(0, 1))) / np.std(gfs_target_data,
                                                                                             axis=(0, 1))

        assert len(self.synop_data_indices) == len(
            all_gfs_target_data), f"len(all_gfs_target_data) should be {len(self.synop_data_indices)} but was {len(all_gfs_target_data)}"

        assert len(self.synop_data_indices) == len(
            all_gfs_input_data), f"len(all_gfs_input_data) should be {len(self.synop_data_indices)} but was {len(all_gfs_input_data)}"
        return self.synop_data_indices, all_gfs_input_data, gfs_target_data, all_gfs_target_data

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

    def prepare_gfs_data_with_wind_components(self, gfs_data: np.ndarray):
        gfs_data = np.delete(gfs_data, self.gfs_wind_components_indices, -1)
        velocity, sin, cos = extend_wind_components(gfs_data[:, :, self.gfs_wind_components_indices])
        gfs_data = normalize_gfs_data(np.concatenate([gfs_data, np.expand_dims(velocity, -1)], -1), self.normalization_type, (0, 1))
        gfs_data = np.concatenate([gfs_data, np.expand_dims(sin, -1), np.expand_dims(cos, -1)], -1)
        return gfs_data

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, x: List[Tuple]):
        variables, dates = [item[:-2] for item in x], [item[-2:] for item in x]
        all_data = [*default_collate(variables), *list(zip(*dates))]
        dict_data = {
            BatchKeys.SYNOP_PAST_Y.value: all_data[0],
            BatchKeys.SYNOP_PAST_X.value: all_data[1],
            BatchKeys.SYNOP_FUTURE_Y.value: all_data[2],
            BatchKeys.SYNOP_FUTURE_X.value: all_data[3]
        }

        if self.config.experiment.use_gfs_data:
            dict_data[BatchKeys.GFS_PAST_X.value] = all_data[4]
            dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[5]
            dict_data[BatchKeys.GFS_FUTURE_X.value] = all_data[6]
            dict_data[BatchKeys.DATES_PAST.value] = all_data[7]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[8]

        else:
            dict_data[BatchKeys.DATES_PAST.value] = all_data[4]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[5]
        return dict_data

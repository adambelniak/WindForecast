import os
from typing import Optional, List, Tuple

import pandas as pd
import wandb
from torch.utils.data import DataLoader, default_collate

from gfs_oper.common import Config as GFSConfig
from gfs_oper.gfs_fetch_job import fetch_oper_gfs
from synop.consts import SYNOP_PERIODIC_FEATURES, LOWER_CLOUDS, CLOUD_COVER
from synop.fetch_oper import fetch_recent_synop
from util.coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.Sequence2SequenceGFSDataset import Sequence2SequenceGFSDataset
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.preprocess.synop.synop_preprocess import modify_feature_names_after_periodic_reduction, \
    decompose_periodic_features
from wind_forecast.util.common_util import NormalizationType
from wind_forecast.util.config import process_config
from wind_forecast.util.df_util import normalize_data_for_test
from wind_forecast.util.gfs_util import add_param_to_train_params, get_gfs_target_param, extend_wind_components

"""
Used for prediction - fetches operational gfs forecasts and synop observations
"""


class Sequence2SequenceOperDataModule(SplittableDataModule):
    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)
        if not config.experiment.use_pretrained_artifact:
            raise Exception(
                "config.experiment.use_pretrained_artifact must be True to use trained model for predictions")

        self.config = config
        self.batch_size = 1

        self.synop_train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        all_params = add_param_to_train_params(self.synop_train_params, self.target_param)
        self.synop_feature_names = list(list(zip(*all_params))[1])

        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.normalization_type = config.experiment.normalization_type
        self.prediction_offset = config.experiment.prediction_offset
        coords = config.experiment.target_coords
        self.target_coords = Coords(coords[0], coords[0], coords[1], coords[1])

        self.gfs_features_params = process_config(config.experiment.train_parameters_config_file).params
        self.gfs_features_names = [f"{f['name']}_{f['level']}" for f in self.gfs_features_params]
        self.gfs_wind_parameters = ["V GRD_HTGL_10", "U GRD_HTGL_10"]

        self.gfs_target_param = get_gfs_target_param(self.target_param)

        self.periodic_features = config.experiment.synop_periodic_features
        self.uses_future_sequences = True

        self.synop_mean = ...
        self.synop_std = ...
        self.synop_min = ...
        self.synop_max = ...
        self.gfs_mean = ...
        self.gfs_std = ...
        self.gfs_min = ...
        self.gfs_max = ...
        self.set_normalization_values()

        self.synop_data = ...
        self.gfs_data = ...
        self.data_indices = [0]

    def prepare_data(self, *args, **kwargs):
        if self.config.experiment.load_gfs_data:
            self.prepare_gfs()

        self.synop_data = fetch_recent_synop(self.config.experiment.synop_station_code,
                                             self.config.experiment.tele_station_code)
        self.synop_data = self.synop_data.tail(self.sequence_length).reset_index()

        self.synop_data = decompose_periodic_features(self.synop_data, self.synop_feature_names)

        features_to_normalize = [name for name in self.synop_feature_names if name not in
                                 modify_feature_names_after_periodic_reduction(
                                     [f['column'][1] for f in SYNOP_PERIODIC_FEATURES])]
        self.synop_data = normalize_data_for_test(
            self.synop_data, features_to_normalize, self.synop_mean, self.synop_std,
            self.normalization_type)

    def setup(self, stage: Optional[str] = None):
        if self.config.experiment.load_gfs_data:
            synop_dataset = SequenceDataset(self.config, self.synop_data, self.data_indices, self.synop_feature_names)
            synop_dataset.set_mean(self.synop_mean)
            synop_dataset.set_std(self.synop_std)
            synop_dataset.set_min(self.synop_min)
            synop_dataset.set_max(self.synop_max)

            gfs_dataset = Sequence2SequenceGFSDataset(self.config, self.gfs_data, self.data_indices, self.gfs_features_names)
            gfs_dataset.set_mean(self.gfs_mean)
            gfs_dataset.set_std(self.gfs_std)
            gfs_dataset.set_min(self.gfs_min)
            gfs_dataset.set_max(self.gfs_max)
            self.dataset_predict = ConcatDatasets(synop_dataset, gfs_dataset)
        else:
            synop_dataset = SequenceDataset(self.config, self.synop_data, self.data_indices, self.synop_feature_names)
            synop_dataset.set_mean(self.synop_mean)
            synop_dataset.set_std(self.synop_std)
            synop_dataset.set_min(self.synop_min)
            synop_dataset.set_max(self.synop_max)
            self.dataset_predict = synop_dataset

        if len(self.dataset_predict) == 0:
            raise RuntimeError("There are no valid samples in the dataset! Please check your run configuration")

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=1, collate_fn=self.collate_fn)

    def collate_fn(self, x: List[Tuple]):
        if self.config.experiment.load_gfs_data:
            synop_data, gfs_data = [item[0] for item in x], [item[1] for item in x]
            synop_data, dates = [[*item[:2], *item[4:]] for item in synop_data], [item[2:4] for item in synop_data]
            all_data = [*default_collate(synop_data), *default_collate(gfs_data),  *list(zip(*dates))]
        else:
            synop_data, dates = [[*item[:2], *item[4:]] for item in x], [item[2:4] for item in x]
            all_data = [*default_collate(synop_data),  *list(zip(*dates))]

        dict_data = {
            BatchKeys.SYNOP_PAST_Y.value: all_data[0],
            BatchKeys.SYNOP_PAST_X.value: all_data[1],
            BatchKeys.DATES_PAST.value: all_data[-2],
            BatchKeys.DATES_FUTURE.value: all_data[-1]
        }

        if self.config.experiment.load_gfs_data:
            dict_data[BatchKeys.GFS_PAST_X.value] = all_data[2]
            dict_data[BatchKeys.GFS_PAST_Y.value] = all_data[3]
            dict_data[BatchKeys.GFS_FUTURE_X.value] = all_data[4]
            dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[5]

        return dict_data

    def prepare_gfs_data_with_wind_components(self, gfs_data: pd.DataFrame):
        gfs_wind_data = gfs_data[self.gfs_wind_parameters]
        gfs_data.drop(columns=self.gfs_wind_parameters, inplace=True)
        velocity, sin, cos = extend_wind_components(gfs_wind_data.values)
        gfs_data["wind-velocity"] = velocity
        gfs_data["wind-sin"] = sin
        gfs_data["wind-cos"] = cos
        self.gfs_features_names.remove(self.gfs_wind_parameters[0])
        self.gfs_features_names.remove(self.gfs_wind_parameters[1])
        self.gfs_features_names.extend(["wind-velocity", "wind-sin", "wind-cos"])
        return gfs_data

    def set_normalization_values(self):
        api = wandb.Api()
        entity = os.getenv('WANDB_ENTITY', '')
        project = os.getenv('WANDB_PROJECT', '')
        run_id = self.config.experiment.prediction_meta_run
        wandb_run = api.run(f"{entity}/{project}/{run_id}")
        history = wandb_run.history()

        if self.normalization_type == NormalizationType.STANDARD:
            synop_mean_dict = history[[col for col in history.columns if col.startswith('synop_mean.')]].iloc[
                -1].to_dict()
            synop_std_dict = history[[col for col in history.columns if col.startswith('synop_std.')]].iloc[
                -1].to_dict()

            self.synop_mean = {key.replace('synop_mean.', ''): v for key, v in synop_mean_dict.items()}
            self.synop_std = {key.replace('synop_std.', ''): v for key, v in synop_std_dict.items()}

            if self.config.experiment.load_gfs_data:
                gfs_mean_dict = history[[col for col in history.columns if col.startswith('gfs_mean.')]].iloc[
                    -1].to_dict()
                gfs_std_dict = history[[col for col in history.columns if col.startswith('gfs_std.')]].iloc[
                    -1].to_dict()

                self.gfs_mean = {key.replace('gfs_mean.', ''): v for key, v in gfs_mean_dict.items()}
                self.gfs_std = {key.replace('gfs_std.', ''): v for key, v in gfs_std_dict.items()}

        else:
            synop_min_dict = history[[col for col in history.columns if col.startswith('synop_min.')]].iloc[
                -1].to_dict()
            synop_max_dict = history[[col for col in history.columns if col.startswith('synop_max.')]].iloc[
                -1].to_dict()

            self.synop_min = {key.replace('synop_min.', ''): v for key, v in synop_min_dict.items()}
            self.synop_max = {key.replace('synop_max.', ''): v for key, v in synop_max_dict.items()}

            if self.config.experiment.load_gfs_data:
                gfs_min_dict = history[[col for col in history.columns if col.startswith('gfs_min.')]].iloc[
                    -1].to_dict()
                gfs_max_dict = history[[col for col in history.columns if col.startswith('gfs_max.')]].iloc[
                    -1].to_dict()

                self.gfs_min = {key.replace('gfs_min.', ''): v for key, v in gfs_min_dict.items()}
                self.gfs_max = {key.replace('gfs_max.', ''): v for key, v in gfs_max_dict.items()}

    def prepare_gfs(self):
        gfs_config = GFSConfig(self.config.experiment.sequence_length - 5,
                               self.config.experiment.future_sequence_length + 5,
                               "download", "processed", self.target_coords)
        self.gfs_data = fetch_oper_gfs(gfs_config)
        if self.gfs_data is None:
            raise Exception("GFS forecasts unavailable :(")

        if all([f in self.gfs_features_names for f in self.gfs_wind_parameters]):
            self.gfs_data = self.prepare_gfs_data_with_wind_components(self.gfs_data)

        features_to_normalize = [name for name in self.gfs_features_names if
                                 name not in ["wind-sin", "wind-cos", self.gfs_target_param]]

        if self.normalization_type == NormalizationType.STANDARD:
            self.gfs_data = normalize_data_for_test(self.gfs_data, features_to_normalize, self.gfs_mean, self.gfs_std,
                                                    self.normalization_type)
        else:
            self.gfs_data = normalize_data_for_test(self.gfs_data, features_to_normalize, self.gfs_min, self.gfs_max,
                                                    self.normalization_type)

        target_data = pd.to_numeric(self.gfs_data[self.gfs_target_param])

        if self.target_param in [LOWER_CLOUDS[1], CLOUD_COVER[1]]:
            self.gfs_data[self.gfs_target_param] = target_data / 100 * (8/9)
        else:
            if self.normalization_type == NormalizationType.STANDARD:
                self.gfs_data[self.gfs_target_param] = (target_data - self.gfs_mean[self.gfs_target_param]) / \
                                                       self.gfs_std[self.gfs_target_param]
            else:
                self.gfs_data[self.gfs_target_param] = (target_data - self.gfs_min[self.gfs_target_param]) / (
                        self.gfs_max[self.gfs_target_param] - self.gfs_min[self.gfs_target_param])

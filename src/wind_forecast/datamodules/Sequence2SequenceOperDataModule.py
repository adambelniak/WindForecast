import torch

from gfs_oper.gfs_fetch_job import fetch_oper_gfs
from synop.fetch_oper import fetch_recent_synop
from util.coords import Coords
from wind_forecast.config.register import Config
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.util.common_util import get_pretrained_artifact_path
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import add_param_to_train_params, get_gfs_target_param, GFSUtil
from gfs_oper.common import Config as GFSConfig

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

        self.gfs_util = GFSUtil(self.target_coords, self.sequence_length, self.future_sequence_length,
                                self.prediction_offset, self.gfs_features_params)

        self.periodic_features = config.experiment.synop_periodic_features
        self.uses_future_sequences = True

        self.synop_data = ...
        self.gfs_data = ...
        self.data_indices = ...
        self.synop_mean = ...
        self.synop_std = ...
        self.synop_dates = ...

    def prepare_data(self, *args, **kwargs):
        if self.config.experiment.load_gfs_data:
            gfs_config = GFSConfig(self.config.experiment.sequence_length,
                                   self.config.experiment.future_sequence_length,
                                   "download", "processed", self.target_coords)
            self.gfs_data = fetch_oper_gfs(gfs_config)
            if self.gfs_data is None:
                raise Exception("GFS forecasts unavailable :(")

        self.synop_data = fetch_recent_synop(self.config.experiment.synop_station_code,
                                             self.config.experiment.tele_station_code)


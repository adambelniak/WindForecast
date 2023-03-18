from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.datamodules.Sequence2SequenceDataModule import Sequence2SequenceDataModule
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.Sequence2SequenceGFSDataset import Sequence2SequenceGFSDataset
from wind_forecast.datasets.Sequence2SequenceSynopDataset import Sequence2SequenceSynopDataset
from wind_forecast.loaders.CMAXLoader import CMAXLoader
from wind_forecast.util.cmax_util import get_available_cmax_hours, \
    initialize_synop_dates_for_sequence_with_cmax, get_mean_and_std_cmax, get_min_max_cmax
from wind_forecast.util.common_util import NormalizationType


class Sequence2SequenceWithCMAXDataModule(Sequence2SequenceDataModule):
    def __init__(self, config: Config):
        super().__init__(config)
        self.load_future_cmax = config.experiment.load_future_cmax
        self.cmax_from_year = config.experiment.cmax_from_year
        self.cmax_to_year = config.experiment.cmax_to_year
        self.dim = config.experiment.cmax_sample_size
        self.cmax_available_ids = ...

    def after_synop_loaded(self):
        self.cmax_available_ids = get_available_cmax_hours(from_year=self.cmax_from_year, to_year=self.cmax_to_year)

        self.synop_dates = initialize_synop_dates_for_sequence_with_cmax(self.cmax_available_ids,
                                                                         self.synop_data,
                                                                         self.sequence_length,
                                                                         self.future_sequence_length,
                                                                         self.prediction_offset,
                                                                         self.load_future_cmax)
        self.synop_data = self.synop_data.reset_index()

    def setup(self, stage: Optional[str] = None):
        if not self.config.experiment.load_cmax_data:
            super().setup(stage)
        if self.initialized:
            self.log_dataset_info()
            return
        if self.get_from_cache(stage):
            self.log_dataset_info()
            return

        cmax_loader = CMAXLoader()

        if self.config.experiment.load_gfs_data:
            self.prepare_dataset_for_gfs()
            synop_dataset = Sequence2SequenceSynopDataset(self.config, self.synop_data, self.data_indices,
                                                          self.synop_feature_names)
            synop_dataset.set_mean(self.synop_mean)
            synop_dataset.set_std(self.synop_std)
            synop_dataset.set_min(self.synop_min)
            synop_dataset.set_max(self.synop_max)

            gfs_dataset = Sequence2SequenceGFSDataset(self.config, self.gfs_data, self.data_indices,
                                                      self.gfs_features_names)
            gfs_dataset.set_mean(self.gfs_mean)
            gfs_dataset.set_std(self.gfs_std)
            gfs_dataset.set_min(self.gfs_min)
            gfs_dataset.set_max(self.gfs_max)

            self.synop_dates = self.synop_data.loc[self.data_indices]['date'].values

            mean, std, min, max = self.get_cmax_normalization_values(cmax_loader)
            cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates,
                                       cmax_values=cmax_loader.get_all_loaded_cmax_images(),
                                       normalize=True, use_future_values=self.load_future_cmax)
            cmax_dataset.set_mean(mean)
            cmax_dataset.set_std(std)
            cmax_dataset.set_min(min)
            cmax_dataset.set_max(max)
            assert len(synop_dataset) == len(cmax_dataset), \
                f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
            dataset = ConcatDatasets(synop_dataset, gfs_dataset, cmax_dataset)
        else:
            synop_dataset = Sequence2SequenceSynopDataset(self.config, self.synop_data, self.data_indices,
                                                          self.synop_feature_names)
            synop_dataset.set_mean(self.synop_mean)
            synop_dataset.set_std(self.synop_std)
            synop_dataset.set_min(self.synop_min)
            synop_dataset.set_max(self.synop_max)

            self.synop_dates = self.synop_data.loc[self.data_indices]['date'].values
            mean, std, min, max = self.get_cmax_normalization_values(cmax_loader)
            cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates,
                                       cmax_values=cmax_loader.get_all_loaded_cmax_images(),
                                       normalize=True, use_future_values=self.load_future_cmax)
            cmax_dataset.set_mean(mean)
            cmax_dataset.set_std(std)
            cmax_dataset.set_min(min)
            cmax_dataset.set_max(max)
            assert len(synop_dataset) == len(
                cmax_dataset), f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
            dataset = ConcatDatasets(synop_dataset, cmax_dataset)

        self.split_dataset(self.config, dataset, self.sequence_length)
        self.log_dataset_info()

        if self.config.experiment._tags_[0] == 'GFS':
            self.eliminate_gfs_bias()

    def collate_fn(self, x: List[Tuple]):
        if not self.config.experiment.load_cmax_data:
            return super().collate_fn(x)

        s2s_data, cmax_data = [item[0] for item in x], [item[1] for item in x]
        variables, dates = [item[:-2] for item in s2s_data], [item[-2:] for item in s2s_data]
        if self.load_future_cmax:
            all_data = [*default_collate(variables), *list(zip(*dates)),
                        *default_collate(np.array(cmax_data)).permute((1, 0, 2, 3, 4))]
        else:
            all_data = [*default_collate(variables), *list(zip(*dates)), torch.Tensor(np.array(cmax_data))]

        dict_data = {BatchKeys.SYNOP_PAST_Y.value: all_data[0], BatchKeys.SYNOP_PAST_X.value: all_data[1],
                     BatchKeys.SYNOP_FUTURE_Y.value: all_data[2], BatchKeys.SYNOP_FUTURE_X.value: all_data[3]}

        if self.load_future_cmax:
            dict_data[BatchKeys.CMAX_PAST.value] = all_data[-2]
            dict_data[BatchKeys.CMAX_FUTURE.value] = all_data[-1]
        else:
            dict_data[BatchKeys.CMAX_PAST.value] = all_data[-1]

        if self.config.experiment.load_gfs_data:
            dict_data[BatchKeys.GFS_PAST_X.value] = all_data[4]
            dict_data[BatchKeys.GFS_PAST_Y.value] = all_data[5]
            dict_data[BatchKeys.GFS_FUTURE_X.value] = all_data[6]
            dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[7]
            dict_data[BatchKeys.DATES_PAST.value] = all_data[8]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[9]
            if self.config.experiment.differential_forecast:
                target_mean = self.dataset_train.dataset.get_dataset("Sequence2SequenceSynopDataset").mean[self.target_param]
                target_std = self.dataset_train.dataset.get_dataset("Sequence2SequenceSynopDataset").std[self.target_param]

                gfs_past_y = dict_data[BatchKeys.GFS_PAST_Y.value] * target_std + target_mean
                gfs_future_y = dict_data[BatchKeys.GFS_FUTURE_Y.value] * target_std + target_mean
                synop_past_y = dict_data[BatchKeys.SYNOP_PAST_Y.value].unsqueeze(-1) * target_std + target_mean
                synop_future_y = dict_data[BatchKeys.SYNOP_FUTURE_Y.value].unsqueeze(-1) * target_std + target_mean
                diff_past = gfs_past_y - synop_past_y
                diff_future = gfs_future_y - synop_future_y
                dict_data[BatchKeys.GFS_SYNOP_PAST_DIFF.value] = diff_past / target_std
                dict_data[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value] = diff_future / target_std

        else:
            dict_data[BatchKeys.DATES_PAST.value] = all_data[4]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[5]
        return dict_data

    def get_cmax_normalization_values(self, cmax_loader: CMAXLoader):
        if self.config.experiment.cmax_normalization_type == NormalizationType.STANDARD:
            if self.load_future_cmax:
                mean, std = get_mean_and_std_cmax(cmax_loader, self.synop_dates, self.dim,
                                                  self.sequence_length,
                                                  self.future_sequence_length,
                                                  self.prediction_offset)
            else:
                mean, std = get_mean_and_std_cmax(cmax_loader, self.synop_dates, self.dim,
                                                  self.sequence_length)

            return mean, std, None, None
        else:
            if self.load_future_cmax:
                min, max = get_min_max_cmax(cmax_loader, self.synop_dates, self.sequence_length,
                                            self.future_sequence_length,
                                            self.prediction_offset)
            else:
                min, max = get_min_max_cmax(cmax_loader, self.synop_dates, self.sequence_length)
            return None, None, min, max

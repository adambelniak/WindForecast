from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.datamodules.Sequence2SequenceDataModule import Sequence2SequenceDataModule
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.Sequence2SequenceDataset import Sequence2SequenceDataset
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset
from wind_forecast.util.cmax_util import get_available_cmax_hours, \
    initialize_synop_dates_for_sequence_with_cmax, CMAX_MIN, CMAX_MAX


class Sequence2SequenceWithCMAXDataModule(Sequence2SequenceDataModule):
    def __init__(self, config: Config):
        super().__init__(config)
        self.load_future_cmax = config.experiment.load_future_cmax
        self.cmax_from_year = config.experiment.cmax_from_year
        self.cmax_to_year = config.experiment.cmax_to_year
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
        if self.initialized:
            return
        if self.get_from_cache(stage):
            return

        if self.config.experiment.load_gfs_data:
            self.prepare_dataset_for_gfs()
            self.synop_dates = self.synop_data.loc[self.data_indices]['date'].values
            synop_dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.gfs_data, self.data_indices,
                                                            self.synop_feature_names, self.gfs_features_names)
            if self.config.experiment.load_cmax_data:
                cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates, normalize=True,
                                           use_future_values=self.load_future_cmax)
                assert len(synop_dataset) == len(cmax_dataset),\
                    f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
                dataset = ConcatDatasets(synop_dataset, cmax_dataset)
            else:
                dataset = synop_dataset
        else:
            synop_dataset = Sequence2SequenceDataset(self.config, self.synop_data, self.data_indices,
                                                     self.synop_feature_names)
            if self.config.experiment.load_cmax_data:
                cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates, normalize=True, use_future_values=self.load_future_cmax)
                assert len(synop_dataset) == len(
                    cmax_dataset), f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
                dataset = ConcatDatasets(synop_dataset, cmax_dataset)
            else:
                dataset = synop_dataset

        if self.config.experiment.load_cmax_data:
            dataset.set_mean([self.synop_mean, 0])
            dataset.set_std([self.synop_std, 0])
            dataset.set_min([0, CMAX_MIN])
            dataset.set_max([0, CMAX_MAX])
        else:
            dataset.set_mean(self.synop_mean)
            dataset.set_std(self.synop_std)

        self.split_dataset(self.config, dataset, self.sequence_length)
        if self.config.experiment._tags_[0] == 'GFS':
            self.eliminate_gfs_bias()

    def collate_fn(self, x: List[Tuple]):
        if not self.config.experiment.load_cmax_data:
            return super().collate_fn(x)

        s2s_data, cmax_data = [item[0] for item in x], [item[1] for item in x]
        variables, dates = [item[:-2] for item in s2s_data], [item[-2:] for item in s2s_data]
        if self.load_future_cmax:
            all_data = [*default_collate(variables), *list(zip(*dates)), *default_collate(np.array(cmax_data)).permute((1,0,2,3,4))]
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
                if self.config.experiment.load_cmax_data:
                    target_mean = self.dataset_train.dataset.mean[0][self.target_param]
                    target_std = self.dataset_train.dataset.std[0][self.target_param]
                else:
                    target_mean = self.dataset_train.dataset.mean[self.target_param]
                    target_std = self.dataset_train.dataset.std[self.target_param]
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

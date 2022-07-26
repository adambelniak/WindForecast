from typing import Optional, Tuple, List

import torch
from torch.utils.data.dataloader import default_collate

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY, BatchKeys
from wind_forecast.datamodules.Sequence2SequenceDataModule import Sequence2SequenceDataModule
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.Sequence2SequenceDataset import Sequence2SequenceDataset
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import normalize_synop_data_for_training, prepare_synop_dataset
from wind_forecast.util.cmax_util import get_available_cmax_hours, \
    initialize_synop_dates_for_sequence_with_cmax, CMAX_MIN, CMAX_MAX
from wind_forecast.util.logging import log


class Sequence2SequenceWithCMAXDataModule(Sequence2SequenceDataModule):
    def __init__(self, config: Config):
        super().__init__(config)
        self.use_future_cmax = config.experiment.use_future_cmax
        self.cmax_from_year = config.experiment.cmax_from_year
        self.cmax_to_year = config.experiment.cmax_to_year
        self.cmax_available_ids = ...
        self.synop_dates = ...

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

        if self.config.debug_mode:
            self.synop_data = self.synop_data.head(self.sequence_length * 20)

        self.cmax_available_ids = get_available_cmax_hours(from_year=self.cmax_from_year, to_year=self.cmax_to_year)

        self.synop_dates = initialize_synop_dates_for_sequence_with_cmax(self.cmax_available_ids,
                                                                         self.synop_data,
                                                                         self.sequence_length,
                                                                         self.future_sequence_length,
                                                                         self.prediction_offset,
                                                                         True)

        self.synop_data = self.synop_data.reset_index()

        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(self.synop_dates)].index

        assert len(self.synop_dates) == len(self.synop_data_indices)

        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, self.synop_feature_names, synop_mean, synop_std = normalize_synop_data_for_training(
            self.synop_data, self.synop_data_indices,
            self.feature_names,
            self.sequence_length + self.prediction_offset + self.future_sequence_length,
            self.normalization_type,
            self.periodic_features)

        self.synop_mean = synop_mean[self.target_param]
        self.synop_std = synop_std[self.target_param]
        log.info(f"Synop mean: {synop_mean[self.target_param]}")
        log.info(f"Synop std: {synop_std[self.target_param]}")

    def setup(self, stage: Optional[str] = None):
        if self.initialized:
            return
        if self.get_from_cache(stage):
            return

        if self.config.experiment.load_gfs_data:
            synop_inputs, gfs_past_y, gfs_past_x, gfs_future_y, gfs_future_x = self.prepare_dataset_for_gfs()
            self.synop_dates = self.synop_data.loc[self.synop_data_indices]['date'].values
            synop_dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data, self.synop_data_indices,
                                                            self.synop_feature_names, gfs_future_y, gfs_past_y,
                                                            gfs_future_x, gfs_past_x)
            if self.config.experiment.use_cmax_data:
                cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates, normalize=True)
                assert len(synop_dataset) == len(
                    cmax_dataset), f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
                dataset = ConcatDatasets(synop_dataset, cmax_dataset)
            else:
                dataset = synop_dataset
        else:
            synop_dataset = Sequence2SequenceDataset(self.config, self.synop_data, self.synop_data_indices,
                                                     self.synop_feature_names)
            if self.config.experiment.use_cmax_data:
                cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates, normalize=True)
                assert len(synop_dataset) == len(
                    cmax_dataset), f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
                dataset = ConcatDatasets(synop_dataset, cmax_dataset)
            else:
                dataset = synop_dataset

        dataset.set_mean(self.synop_mean)
        dataset.set_std(self.synop_std)
        dataset.set_min(CMAX_MIN)
        dataset.set_max(CMAX_MAX)
        dataset.set_gfs_std(self.gfs_std)
        dataset.set_gfs_mean(self.gfs_mean)
        self.split_dataset(self.config, dataset, self.sequence_length)

    def collate_fn(self, x: List[Tuple]):
        if not self.config.experiment.use_cmax_data:
            return super().collate_fn(x)

        s2s_data, cmax_data = [item[0] for item in x], [item[1] for item in x]
        variables, dates = [item[:-2] for item in s2s_data], [item[-2:] for item in s2s_data]
        if self.use_future_cmax:
            all_data = [*default_collate(variables), *list(zip(*dates)), *default_collate(cmax_data)]
        else:
            all_data = [*default_collate(variables), *list(zip(*dates)), torch.Tensor(cmax_data)]

        dict_data = {
            BatchKeys.SYNOP_PAST_Y.value: all_data[0],
            BatchKeys.SYNOP_PAST_X.value: all_data[1],
            BatchKeys.SYNOP_FUTURE_Y.value: all_data[2],
            BatchKeys.SYNOP_FUTURE_X.value: all_data[3]
        }

        dict_data[BatchKeys.CMAX_PAST.value] = all_data[-2]
        dict_data[BatchKeys.CMAX_FUTURE.value] = all_data[-1]

        if self.config.experiment.load_gfs_data:
            dict_data[BatchKeys.GFS_PAST_X.value] = all_data[4]
            dict_data[BatchKeys.GFS_PAST_Y.value] = all_data[5]
            dict_data[BatchKeys.GFS_FUTURE_X.value] = all_data[6]
            dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[7]
            dict_data[BatchKeys.DATES_PAST.value] = all_data[8]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[9]
            if self.config.experiment.differential_forecast:
                K_TO_C = 273.15 if self.config.experiment.target_parameter == 'temperature' else 0
                gfs_past_y = dict_data[BatchKeys.GFS_PAST_Y.value] * self.dataset_train.gfs_std + self.dataset_train.gfs_mean - K_TO_C
                gfs_future_y = dict_data[BatchKeys.GFS_FUTURE_Y.value] * self.dataset_train.gfs_std + self.dataset_train.gfs_mean - K_TO_C
                synop_past_y = dict_data[BatchKeys.SYNOP_PAST_Y.value].unsqueeze(-1) * self.dataset_train.std + self.dataset_train.mean
                synop_future_y = dict_data[BatchKeys.SYNOP_FUTURE_Y.value].unsqueeze(-1) * self.dataset_train.std + self.dataset_train.mean
                diff_past = gfs_past_y - synop_past_y
                diff_future = gfs_future_y - synop_future_y
                dict_data[BatchKeys.GFS_SYNOP_PAST_DIFF.value] = diff_past / self.dataset_train.std
                dict_data[BatchKeys.GFS_SYNOP_FUTURE_DIFF.value] = diff_future / self.dataset_train.std

        else:
            dict_data[BatchKeys.DATES_PAST.value] = all_data[4]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[5]
        return dict_data

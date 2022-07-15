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
                                                                         self.use_future_cmax)

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
        print(f"Synop mean: {synop_mean[self.target_param]}")
        print(f"Synop std: {synop_std[self.target_param]}")

    def setup(self, stage: Optional[str] = None):
        if self.initialized:
            return
        if self.get_from_cache(stage):
            return

        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, all_gfs_target_data = self.prepare_dataset_for_gfs()
            self.synop_dates = self.synop_data.loc[self.synop_data_indices]['date'].values
            cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates, normalize=True)
            if self.use_all_gfs_params:
                synop_dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data,
                                                                self.synop_data_indices,
                                                                self.synop_feature_names, gfs_target_data,
                                                                all_gfs_target_data, all_gfs_input_data)
            else:
                synop_dataset = Sequence2SequenceWithGFSDataset(self.config, self.synop_data,
                                                                self.synop_data_indices,
                                                                self.synop_feature_names,
                                                                gfs_target_data)
            assert len(synop_dataset) == len(
                cmax_dataset), f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
            dataset = ConcatDatasets(synop_dataset, cmax_dataset)
        else:
            cmax_dataset = CMAXDataset(config=self.config, dates=self.synop_dates, normalize=True)
            synop_dataset = Sequence2SequenceDataset(self.config, self.synop_data, self.synop_data_indices,
                                                     self.synop_feature_names)
            assert len(synop_dataset) == len(
                cmax_dataset), f"Synop and CMAX datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"

            dataset = ConcatDatasets(synop_dataset, cmax_dataset)

        dataset.set_mean([self.synop_mean, 0])
        dataset.set_std([self.synop_std, 0])
        dataset.set_min([0, CMAX_MIN])
        dataset.set_max([0, CMAX_MAX])
        self.split_dataset(self.config, dataset, self.sequence_length)

    def collate_fn(self, x: List[Tuple]):
        s2s_data, cmax_data = [item[0] for item in x], [item[1] for item in x]
        tensors, dates = [item[:-2] for item in s2s_data], [item[-2:] for item in s2s_data]
        if self.use_future_cmax:
            all_data = [*default_collate(tensors), *list(zip(*dates)), *default_collate(cmax_data)]
        else:
            all_data = [*default_collate(tensors), *list(zip(*dates)), torch.Tensor(cmax_data)]

        dict_data = {
            BatchKeys.SYNOP_PAST_Y.value: all_data[0],
            BatchKeys.SYNOP_PAST_X.value: all_data[1],
            BatchKeys.SYNOP_FUTURE_Y.value: all_data[2],
            BatchKeys.SYNOP_FUTURE_X.value: all_data[3]
        }

        if self.config.experiment.use_future_cmax:
            dict_data[BatchKeys.CMAX_PAST.value] = all_data[-2]
            dict_data[BatchKeys.CMAX_TARGETS.value] = all_data[-1]
        else:
            dict_data[BatchKeys.CMAX_PAST.value] = all_data[-1]

        if self.config.experiment.use_gfs_data:
            if self.use_all_gfs_params:
                dict_data[BatchKeys.GFS_PAST_X.value] = all_data[4]
                dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[5]
                dict_data[BatchKeys.GFS_FUTURE_X.value] = all_data[6]
                dict_data[BatchKeys.DATES_PAST.value] = all_data[7]
                dict_data[BatchKeys.DATES_FUTURE.value] = all_data[8]

            else:
                dict_data[BatchKeys.GFS_FUTURE_Y.value] = all_data[4]
                dict_data[BatchKeys.DATES_PAST.value] = all_data[5]
                dict_data[BatchKeys.DATES_FUTURE.value] = all_data[6]

        else:
            dict_data[BatchKeys.DATES_PAST.value] = all_data[4]
            dict_data[BatchKeys.DATES_FUTURE.value] = all_data[5]
        return dict_data

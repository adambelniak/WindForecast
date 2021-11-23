import os
from typing import Optional, Tuple, List

from torch.utils.data.dataloader import default_collate

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY, BatchKeys
from wind_forecast.datamodules.DataModulesCache import DataModulesCache
from wind_forecast.datamodules.Sequence2SequenceDataModule import Sequence2SequenceDataModule
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.Sequence2SequenceDataset import Sequence2SequenceDataset
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import normalize_synop_data, prepare_synop_dataset
from wind_forecast.util.cmax_util import get_available_cmax_hours, \
    initialize_CMAX_list_IDs_and_synop_dates_for_sequence
from wind_forecast.util.common_util import split_dataset


class Sequence2SequenceWithCMAXDataModule(Sequence2SequenceDataModule):
    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)
        self.use_future_cmax = config.experiment.use_future_cmax
        self.cmax_from_year = config.experiment.cmax_from_year
        self.cmax_to_year = config.experiment.cmax_to_year
        self.cmax_IDs = ...

    def prepare_data(self, *args, **kwargs):
        self.synop_data = prepare_synop_dataset(self.synop_file,
                                                list(list(zip(*self.train_params))[1]),
                                                dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                from_year=self.synop_from_year,
                                                to_year=self.synop_to_year,
                                                norm=False)

        if os.getenv('RUN_MODE', '').lower() == 'debug':
            self.synop_data = self.synop_data.head(100)

        available_ids = get_available_cmax_hours(from_year=self.cmax_from_year,
                                                 to_year=self.cmax_to_year)

        self.cmax_IDs, dates = initialize_CMAX_list_IDs_and_synop_dates_for_sequence(available_ids,
                                                                                     self.synop_data,
                                                                                     self.sequence_length,
                                                                                     self.future_sequence_length,
                                                                                     self.prediction_offset,
                                                                                     self.use_future_cmax)

        self.synop_data = self.synop_data.reset_index()

        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index

        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, synop_mean, synop_std = normalize_synop_data(self.synop_data, self.synop_data_indices,
                                                                      self.feature_names,
                                                                      self.sequence_length + self.prediction_offset
                                                                      + self.future_sequence_length,
                                                                      self.normalization_type)
        self.synop_mean = synop_mean[self.target_param_index]
        self.synop_std = synop_std[self.target_param_index]
        print(f"Synop mean: {synop_mean[self.target_param_index]}")
        print(f"Synop std: {synop_std[self.target_param_index]}")

    def setup(self, stage: Optional[str] = None):
        cached_dataset = DataModulesCache().get_cached_dataset()
        if stage == 'test' and cached_dataset is not None:
            self.dataset_test = cached_dataset
            return

        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, all_gfs_target_data = self.prepare_dataset_for_gfs()

            self.cmax_IDs = [item for index, item in enumerate(self.cmax_IDs) if
                             index not in self.removed_dataset_indices]

            assert len(self.cmax_IDs) == len(synop_inputs)

            if self.use_all_gfs_params:
                dataset = ConcatDatasets(Sequence2SequenceWithGFSDataset(self.config, self.synop_data,
                                                                         self.synop_data_indices, gfs_target_data,
                                                                         all_gfs_target_data, all_gfs_input_data),
                                         CMAXDataset(config=self.config, IDs=self.cmax_IDs, normalize=True))
            else:
                dataset = ConcatDatasets(Sequence2SequenceWithGFSDataset(self.config, self.synop_data,
                                                                         self.synop_data_indices, gfs_target_data),
                                         CMAXDataset(config=self.config, IDs=self.cmax_IDs, normalize=True))

        else:
            assert len(self.cmax_IDs) == len(self.synop_data_indices)

            dataset = ConcatDatasets(
                Sequence2SequenceDataset(self.config, self.synop_data, self.synop_data_indices),
                CMAXDataset(config=self.config, IDs=self.cmax_IDs, normalize=True))

        dataset.set_mean([self.synop_mean, 0])
        dataset.set_std([self.synop_std, 0])
        self.dataset_train, self.dataset_val = split_dataset(dataset, self.config.experiment.val_split,
                                                             sequence_length=self.sequence_length if self.sequence_length > 1 else None)
        self.dataset_test = self.dataset_val
        DataModulesCache().cache_dataset(self.dataset_test)

    def collate_fn(self, x: List[Tuple]):
        s2s_data, cmax_data = [item[0] for item in x], [item[1] for item in x]
        tensors, dates = [item[:-2] for item in s2s_data], [item[-2:] for item in s2s_data]
        all_data = [*default_collate(tensors), *list(zip(*dates)), *default_collate(cmax_data)]
        dict_data = {
            BatchKeys.SYNOP_INPUTS.value: all_data[0],
            BatchKeys.SYNOP_TARGETS.value: all_data[1],
            BatchKeys.ALL_SYNOP_TARGETS.value: all_data[2],
            BatchKeys.CMAX_INPUTS.value: all_data[-2],
            BatchKeys.CMAX_TARGETS.value: all_data[-1]
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
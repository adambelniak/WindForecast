from typing import Optional

from torch.utils.data import DataLoader

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.SequenceDataModule import SequenceDataModule
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.datasets.SequenceWithGFSDataset import SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset, normalize_synop_data_for_training
from wind_forecast.util.cmax_util import get_available_cmax_hours, \
    initialize_CMAX_list_IDs_and_synop_dates_for_sequence


class SequenceWithCMAXDataModule(SequenceDataModule):
    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)

        self.cmax_from_year = config.experiment.cmax_from_year
        self.cmax_to_year = config.experiment.cmax_to_year
        self.cmax_IDs = ...

    def prepare_data(self, *args, **kwargs):
        self.synop_data = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]),
                                                dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                from_year=self.synop_from_year,
                                                to_year=self.synop_to_year,
                                                norm=False)

        available_ids = get_available_cmax_hours(from_year=self.cmax_from_year,
                                                 to_year=self.cmax_to_year)

        self.cmax_IDs, dates = initialize_CMAX_list_IDs_and_synop_dates_for_sequence(available_ids,
                                                                                     self.synop_data,
                                                                                     self.sequence_length,
                                                                                     1,
                                                                                     self.prediction_offset)

        self.synop_data = self.synop_data.reset_index()
        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index
        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, self.synop_feature_names, synop_mean, synop_std = normalize_synop_data_for_training(self.synop_data, self.synop_data_indices,
                                                                                                             self.feature_names,
                                                                                                             self.sequence_length + self.prediction_offset,
                                                                                                             self.normalization_type)
        print(f"Synop mean: {synop_mean[self.target_param]}")
        print(f"Synop std: {synop_std[self.target_param]}")

    def setup(self, stage: Optional[str] = None):
        if self.get_from_cache(stage):
            return
        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, synop_targets = self.prepare_dataset_for_gfs()

            self.cmax_IDs = [item for index, item in enumerate(self.cmax_IDs) if
                             index not in self.removed_dataset_indices]

            assert len(self.cmax_IDs) == len(synop_inputs)
            if self.gfs_train_params is not None:
                dataset = ConcatDatasets(SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets, all_gfs_input_data),
                                         CMAXDataset(config=self.config, IDs=self.cmax_IDs, normalize=True))
            else:
                dataset = ConcatDatasets(SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets),
                                         CMAXDataset(config=self.config, IDs=self.cmax_IDs, normalize=True))
        else:
            assert len(self.cmax_IDs) == len(self.synop_data_indices)
            dataset = ConcatDatasets(SequenceDataset(config=self.config, synop_data=self.synop_data, synop_data_indices=self.synop_data_indices),
                                     CMAXDataset(config=self.config, IDs=self.cmax_IDs, normalize=True))

        self.split_dataset(dataset, self.sequence_length)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

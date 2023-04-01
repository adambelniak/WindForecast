from typing import Optional

from torch.utils.data import DataLoader

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.SequenceDataModule import SequenceDataModule
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.datasets.SequenceWithGFSDataset import SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.cmax_util import get_available_cmax_hours, \
    initialize_synop_dates_for_sequence_with_cmax
from wind_forecast.util.df_util import normalize_data_for_training
from wind_forecast.util.logging import log


class SequenceWithCMAXDataModule(SequenceDataModule):
    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)

        self.cmax_from_year = config.experiment.cmax_from_year
        self.cmax_to_year = config.experiment.cmax_to_year
        self.cmax_available_ids = ...
        self.synop_dates = ...

    def prepare_data(self, *args, **kwargs):
        self.synop_data = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]),
                                                dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                from_year=self.synop_from_year,
                                                to_year=self.synop_to_year,
                                                norm=False)

        self.cmax_available_ids = get_available_cmax_hours(from_year=self.cmax_from_year,
                                                           to_year=self.cmax_to_year)

        self.synop_dates = initialize_synop_dates_for_sequence_with_cmax(self.cmax_available_ids,
                                                                         self.synop_data,
                                                                         self.sequence_length,
                                                                         1,
                                                                         self.prediction_offset)

        self.synop_data = self.synop_data.reset_index()
        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        self.synop_data_indices = self.synop_data[self.synop_data["date"].isin(self.synop_dates)].index
        # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
        self.synop_data, self.synop_mean, self.synop_std, min, max = normalize_data_for_training(
            self.synop_data, self.synop_data_indices,
            self.feature_names,
            self.sequence_length + self.prediction_offset,
            self.normalization_type)
        log.info(f"Synop mean: {self.synop_mean[self.target_param]}")
        log.info(f"Synop std: {self.synop_std[self.target_param]}")

    def setup(self, stage: Optional[str] = None):
        if self.get_from_cache(stage):
            return

        cmax_dataset = CMAXDataset(self.config, self.synop_dates, True, False)
        if self.config.experiment.use_gfs_data:
            synop_inputs, all_gfs_input_data, gfs_target_data, synop_targets = self.prepare_dataset_for_gfs()

            if self.gfs_train_params is not None:
                synop_dataset = SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets, all_gfs_input_data)
            else:
                synop_dataset = SequenceWithGFSDataset(synop_inputs, gfs_target_data, synop_targets)

            assert len(synop_dataset) == len(
                cmax_dataset), f"CMAX and synop datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
            dataset = ConcatDatasets(synop_dataset, cmax_dataset)
        else:
            synop_dataset = SequenceDataset(config=self.config, synop_data=self.synop_data,
                                            synop_data_indices=self.synop_data_indices)
            assert len(synop_dataset) == len(
                cmax_dataset), f"CMAX and synop datasets lengths don't match: {len(synop_dataset)} vs {len(cmax_dataset)}"
            dataset = ConcatDatasets(synop_dataset, cmax_dataset)

        self.split_dataset(self.config, dataset, self.sequence_length)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

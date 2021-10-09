from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.cmax_util import get_available_hdf_files_cmax_hours, \
    initialize_CMAX_list_IDs_and_synop_dates_for_sequence
from wind_forecast.util.common_util import split_dataset


class SequenceWithCMAXDataModule(LightningDataModule):
    def __init__(
            self,
            config: Config
    ):
        super().__init__()
        self.config = config
        self.val_split = config.experiment.val_split
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...

        self.sequence_length = config.experiment.sequence_length
        self.synop_file = config.experiment.synop_file
        self.train_params = config.experiment.synop_train_features
        self.labels, _, _ = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]),
                                                  dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                  from_year=config.experiment.synop_from_year,
                                                  to_year=config.experiment.synop_to_year,
                                                  norm=False)

        available_ids = get_available_hdf_files_cmax_hours(from_year=config.experiment.cmax_from_year, to_year=config.experiment.cmax_to_year)
        self.cmax_IDs, self.dates = initialize_CMAX_list_IDs_and_synop_dates_for_sequence(available_ids, self.labels,
                                                                                          self.sequence_length, 1,
                                                                                          config.experiment.prediction_offset)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = ConcatDatasets(
            SequenceDataset(config=self.config, synop_data=self.labels, dates=self.dates),
            CMAXDataset(config=self.config, IDs=self.cmax_IDs, normalize=True))
        self.dataset_train, self.dataset_val = split_dataset(dataset, self.config.experiment.val_split,
                                                             sequence_length=self.sequence_length if self.sequence_length > 1 else None)
        self.dataset_test = self.dataset_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

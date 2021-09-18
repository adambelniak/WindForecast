from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.Sequence2SequenceDataset import Sequence2SequenceDataset
from wind_forecast.datasets.Sequence2SequenceWithGFSDataset import Sequence2SequenceWithGFSDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.cmax_util import get_available_hdf_files_cmax_hours, \
    initialize_CMAX_list_IDs_and_synop_dates_for_sequence


class Sequence2SequenceWithCMAXDataModule(LightningDataModule):
    def __init__(
            self,
            config: Config
    ):
        super().__init__()
        self.config = config
        self.val_split = config.experiment.val_split
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle
        self.use_future_cmax = config.experiment.use_future_cmax
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...

        self.sequence_length = config.experiment.sequence_length
        self.synop_file = config.experiment.synop_file
        self.train_params = config.experiment.synop_train_features

        self.labels, _, _ = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]),
                                                  dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                  from_year=config.experiment.synop_from_year,
                                                  norm=False)

        available_ids = get_available_hdf_files_cmax_hours()
        self.cmax_IDs, self.dates = initialize_CMAX_list_IDs_and_synop_dates_for_sequence(available_ids, self.labels,
                                                                                          self.sequence_length,
                                                                                          config.experiment.future_sequence_length,
                                                                                          config.experiment.prediction_offset,
                                                                                          config.experiment.use_future_cmax)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            if self.config.experiment.use_gfs_data:
                dataset = ConcatDatasets(
                    Sequence2SequenceWithGFSDataset(config=self.config, synop_data=self.labels, dates=self.dates,
                                                    train=True),
                    CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, train=True, normalize=True))
            else:
                dataset = ConcatDatasets(
                    Sequence2SequenceDataset(config=self.config, synop_data=self.labels, dates=self.dates,
                                             train=True),
                    CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, train=True, normalize=True))
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)),
                                                                          int(length * self.val_split)])
        elif stage == 'test':
            if self.config.experiment.use_gfs_data:
                self.dataset_test = \
                    ConcatDatasets(
                        Sequence2SequenceWithGFSDataset(config=self.config, synop_data=self.labels,
                                                        dates=self.dates,
                                                        train=False),
                        CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, train=False, normalize=True))

            else:
                self.dataset_test = ConcatDatasets(
                    Sequence2SequenceDataset(config=self.config, synop_data=self.labels,
                                             dates=self.dates,
                                             train=False),
                    CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, train=False, normalize=True))

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

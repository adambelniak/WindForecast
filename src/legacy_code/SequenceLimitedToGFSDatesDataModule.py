from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datamodules.DataModulesCache import DataModulesCache
from wind_forecast.datasets.SequenceLimitedToGFSDatesDataset import SequenceLimitedToGFSDatesDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.common_util import split_dataset
from wind_forecast.util.synop_util import get_correct_dates_for_sequence


class SequenceLimitedToGFSDatesDataModule(LightningDataModule):

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
        self.synop_file = config.experiment.synop_file
        self.train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        self.sequence_length = config.experiment.sequence_length
        self.labels = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]),
                                            dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                            from_year=config.experiment.synop_from_year,
                                            to_year=config.experiment.synop_to_year,
                                            norm=False)

        self.dates = get_correct_dates_for_sequence(self.labels, self.sequence_length, 1,
                                                    config.experiment.prediction_offset)

    def prepare_data(self, *args, **kwargs):
        pass

    def split_dataset(self, dataset):
        datasets = split_dataset(dataset,
                                 self.config.experiment.val_split,
                                 self.config.experiment.test_split,
                                 split_mode=self.config.experiment.dataset_split_mode,
                                 sequence_length=self.sequence_length if self.sequence_length > 1 else None)
        if self.config.experiment.val_split > 0:
            self.dataset_train, self.dataset_val, self.dataset_test = datasets
        else:
            self.dataset_train, self.dataset_test = datasets
            self.dataset_val = None

        print('Dataset train len: ' + str(len(self.dataset_train)))
        print('Dataset val len: ' + '0' if self.dataset_val is None else str(len(self.dataset_val)))
        print('Dataset test len: ' + str(len(self.dataset_test)))

        DataModulesCache().cache_dataset('dataset_test', self.dataset_test)
        if self.dataset_val is not None:
            DataModulesCache().cache_dataset('dataset_val', self.dataset_val)

    def setup(self, stage: Optional[str] = None):
        if stage == 'test':
            cached_dataset = DataModulesCache().get_cached_dataset('dataset_test')
            if cached_dataset is not None:
                self.dataset_test = cached_dataset
                return

        if stage == 'validate':
            cached_dataset = DataModulesCache().get_cached_dataset('dataset_val')
            if cached_dataset is not None:
                self.dataset_test = cached_dataset
                return
        dataset = SequenceLimitedToGFSDatesDataset(config=self.config, synop_data=self.labels, dates=self.dates)
        self.split_dataset(dataset)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

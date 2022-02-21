from typing import Optional

from pytorch_lightning import LightningDataModule

from wind_forecast.config.register import Config
from wind_forecast.datamodules.DataModulesCache import DataModulesCache
from wind_forecast.util.common_util import split_dataset


class Splittable(LightningDataModule):

    def __init__(self, config: Config):
        super().__init__()
        self.val_split = config.experiment.val_split
        self.test_split = config.experiment.test_split
        self.dataset_split_mode = config.experiment.dataset_split_mode
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...

    def split_dataset(self, dataset, sequence_length):
        datasets = split_dataset(dataset,
                                 self.val_split,
                                 self.test_split,
                                 split_mode=self.dataset_split_mode,
                                 sequence_length=sequence_length if sequence_length > 1 else None)
        if self.val_split > 0:
            self.dataset_train, self.dataset_val, self.dataset_test = datasets
        else:
            self.dataset_train, self.dataset_test = datasets
            self.dataset_val = None

        print('Dataset train len: ' + str(len(self.dataset_train)))
        print('Dataset val len: ' + ('0' if self.dataset_val is None else str(len(self.dataset_val))))
        print('Dataset test len: ' + str(len(self.dataset_test)))

        DataModulesCache().cache_dataset('dataset_test', self.dataset_test)
        if self.dataset_val is not None:
            DataModulesCache().cache_dataset('dataset_val', self.dataset_val)

    def get_from_cache(self, stage: Optional[str] = None):
        if stage == 'test':
            cached_dataset = DataModulesCache().get_cached_dataset('dataset_test')
            if cached_dataset is not None:
                self.dataset_test = cached_dataset
                return True

        if stage == 'validate':
            cached_dataset = DataModulesCache().get_cached_dataset('dataset_val')
            if cached_dataset is not None:
                self.dataset_test = cached_dataset
                return True

        return False
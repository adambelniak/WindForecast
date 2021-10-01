from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader, Subset
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.util.cmax_util import get_available_hdf_files_cmax_hours


class CMAXDataModule(LightningDataModule):

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

        self.cmax_IDs = get_available_hdf_files_cmax_hours(from_year=config.experiment.cmax_from_year)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, normalize=True)

        length = len(dataset)
        seq_length = self.config.experiment.sequence_length
        skip_number_of_frames = (seq_length if seq_length > 1 else 0) + (self.config.experiment.future_sequence_length if self.config.experiment.use_future_cmax else 0)
        self.dataset_train, self.dataset_val = Subset(dataset, np.arange(length - (int(length * self.val_split)))), \
                                               Subset(dataset, np.arange(length - (int(length * self.val_split)) + skip_number_of_frames, length))
        self.dataset_test = self.dataset_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

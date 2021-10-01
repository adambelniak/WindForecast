from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.datasets.SingleGFSPointDataset import SingleGFSPointDataset


class SingleGFSPointDataModule(LightningDataModule):

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

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = SingleGFSPointDataset(config=self.config)
        length = len(dataset)
        seq_length = self.config.experiment.sequence_length
        fut_seq_length = self.config.experiment.future_sequence_length
        skip_number_of_frames = (seq_length if seq_length > 1 else 0) + (
            fut_seq_length if self.config.experiment.future_sequence_length > 1 else 0)
        self.dataset_train, self.dataset_val = Subset(dataset, np.arange(length - (int(length * self.val_split)))), \
                                               Subset(dataset, np.arange(
                                                   length - (int(length * self.val_split)) + skip_number_of_frames,
                                                   int(length * self.val_split)))
        self.dataset_test = self.dataset_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
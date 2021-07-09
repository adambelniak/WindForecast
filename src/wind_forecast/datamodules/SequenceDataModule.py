from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from wind_forecast.config.register import Config
from wind_forecast.datasets.SequenceDataset import SequenceDataset


class SequenceDataModule(LightningDataModule):

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
        self.train_parameters = config.experiment.lstm_train_parameters
        self.prediction_offset = config.experiment.prediction_offset

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            dataset = SequenceDataset(config=self.config, train=True)
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)), int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = SequenceDataset(config=self.config, train=False)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from wind_forecast.config.register import Config
from wind_forecast.datasets.SequenceWithGFSDataset import SequenceWithGFSDataset
from wind_forecast.datasets.SingleGFSPointDataset import SingleGFSPointDataset
from wind_forecast.util.config import process_config
from wind_forecast.util.utils import get_available_numpy_files, target_param_to_gfs_name_level


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
        if stage in (None, 'fit'):
            dataset = SingleGFSPointDataset(config=self.config, train=True)
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)),
                                                                          int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = SingleGFSPointDataset(config=self.config, train=False)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
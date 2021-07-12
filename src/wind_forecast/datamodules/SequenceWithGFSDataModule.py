from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from wind_forecast.config.register import Config
from wind_forecast.datasets.SequenceWithGFSDataset import SequenceWithGFSDataset
from wind_forecast.util.utils import get_available_numpy_files, target_param_to_gfs_name_level


class SequenceWithGFSDataModule(LightningDataModule):

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
        self.gfs_dataset_dir = config.experiment.gfs_dataset_dir
        self.target_param = config.experiment.target_parameter

        self.IDs = get_available_numpy_files(target_param_to_gfs_name_level(self.target_param), self.prediction_offset, self.gfs_dataset_dir)


    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            dataset = SequenceWithGFSDataset(config=self.config, gfs_list_IDs=self.IDs, train=True)
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)), int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = SequenceWithGFSDataset(config=self.config, gfs_list_IDs=self.IDs, train=False)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
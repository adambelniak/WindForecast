from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from wind_forecast.config.register import Config
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.util.cmax_util import get_available_hdf_files_cmax_hours
from wind_forecast.util.common_util import split_dataset


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
        self.sequence_length = self.config.experiment.sequence_length
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...

        self.cmax_IDs = get_available_hdf_files_cmax_hours(from_year=config.experiment.cmax_from_year)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, normalize=True)
        self.dataset_train, self.dataset_val = split_dataset(dataset, self.config.experiment.val_split, sequence_length=self.sequence_length if self.sequence_length > 1 else None)
        self.dataset_test = self.dataset_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

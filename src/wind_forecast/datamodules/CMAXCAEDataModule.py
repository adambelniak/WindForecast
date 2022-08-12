from typing import Optional

from torch.utils.data import DataLoader

from wind_forecast.config.register import Config
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.datasets.CMAXCAEDataset import CMAXCAEDataset
from wind_forecast.util.cmax_util import get_available_cmax_h5


class CMAXCAEDataModule(SplittableDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)
        self.config = config
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle

        self.cmax_IDs = get_available_cmax_h5(from_year=config.experiment.cmax_from_year, to_year=config.experiment.cmax_to_year)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        self.dataset_test = CMAXCAEDataset(self.config, self.cmax_IDs)

    # It will be used only in test phase
    def train_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.config.experiment.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.config.experiment.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.config.experiment.num_workers)

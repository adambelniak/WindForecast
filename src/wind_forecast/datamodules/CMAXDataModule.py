from typing import Optional

from torch.utils.data import DataLoader

from wind_forecast.config.register import Config
from wind_forecast.datamodules.SplittableDataModule import SplittableDataModule
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.util.cmax_util import get_available_cmax_hours, date_from_cmax_date_key


class CMAXDataModule(SplittableDataModule):

    def __init__(
            self,
            config: Config
    ):
        super().__init__(config)
        self.config = config
        self.batch_size = config.experiment.batch_size
        self.shuffle = config.experiment.shuffle
        self.sequence_length = self.config.experiment.sequence_length

        self.cmax_IDs = get_available_cmax_hours(from_year=config.experiment.cmax_from_year, to_year=config.experiment.cmax_to_year)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if self.get_from_cache(stage):
            return
        dataset = CMAXDataset(self.config, [date_from_cmax_date_key(key) for key in self.cmax_IDs], True, False)
        self.split_dataset(self.config, dataset, self.sequence_length)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

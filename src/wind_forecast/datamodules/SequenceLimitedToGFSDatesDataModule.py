from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
import numpy as np
from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.SequenceLimitedToGFSDatesDataset import SequenceLimitedToGFSDatesDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
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
        self.labels, _, _ = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]),
                                                  dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                  from_year=config.experiment.synop_from_year,
                                                  norm=False)

        self.dates = get_correct_dates_for_sequence(self.labels, self.sequence_length, 1,
                                                    config.experiment.prediction_offset)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = SequenceLimitedToGFSDatesDataset(config=self.config, synop_data=self.labels, dates=self.dates)
        length = len(dataset)
        seq_length = self.config.experiment.sequence_length
        skip_number_of_frames = (seq_length if seq_length > 1 else 0)
        self.dataset_train, self.dataset_val = Subset(dataset, np.arange(length - (int(length * self.val_split)))), \
                                               Subset(dataset, np.arange(
                                                   length - (int(length * self.val_split)) + skip_number_of_frames, length))
        self.dataset_test = self.dataset_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

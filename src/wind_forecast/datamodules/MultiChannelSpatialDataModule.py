from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.MultiChannelSpatialDataset import MultiChannelSpatialDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.utils import get_available_numpy_files


class MultiChannelSpatialDataModule(LightningDataModule):

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
        self.train_parameters = process_config(config.experiment.train_parameters_config_file)
        self.prediction_offset = config.experiment.prediction_offset
        self.gfs_dataset_dir = config.experiment.gfs_dataset_dir
        self.synop_file = config.experiment.synop_file
        self.target_param = config.experiment.target_parameter
        self.labels, self.label_mean, self.label_std = prepare_synop_dataset(self.synop_file, [self.target_param],
                                                                             dataset_dir=SYNOP_DATASETS_DIRECTORY)
        self.IDs = get_available_numpy_files(self.train_parameters, self.prediction_offset, self.gfs_dataset_dir)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            dataset = MultiChannelSpatialDataset(config=self.config, train_IDs=self.IDs, labels=self.labels, train=True)
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)), int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = MultiChannelSpatialDataset(config=self.config, train_IDs=self.IDs, labels=self.labels, train=False)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)


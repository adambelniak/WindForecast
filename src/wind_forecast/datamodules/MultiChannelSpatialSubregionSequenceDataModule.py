from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.MultiChannelSpatialSubregionDataset import MultiChannelSpatialSubregionDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.common_util import split_dataset
from wind_forecast.util.config import process_config
from wind_forecast.util.gfs_util import get_available_numpy_files, initialize_GFS_list_IDs_for_sequence, GFS_DATASET_DIR


class MultiChannelSpatialSubregionSequenceDataModule(LightningDataModule):

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
        self.gfs_dataset_dir = GFS_DATASET_DIR
        self.synop_file = config.experiment.synop_file
        self.target_param = config.experiment.target_parameter
        self.sequence_length = config.experiment.sequence_length
        self.labels, self.label_mean, self.label_std = prepare_synop_dataset(self.synop_file, [self.target_param],
                                                                             dataset_dir=SYNOP_DATASETS_DIRECTORY,
                                                                             from_year=config.experiment.synop_from_year,
                                                                             to_year=config.experiment.synop_to_year)
        available_ids = get_available_numpy_files(self.train_parameters, self.prediction_offset, self.gfs_dataset_dir)
        self.IDs = initialize_GFS_list_IDs_for_sequence(available_ids, self.labels, self.train_parameters[0],
                                                        self.target_param,
                                                        self.sequence_length)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        dataset = MultiChannelSpatialSubregionDataset(config=self.config, train_IDs=self.IDs, labels=self.labels,
                                                      normalize=True)
        self.dataset_train, self.dataset_val = split_dataset(dataset, self.config.experiment.val_split, sequence_length=
                                                             self.sequence_length if self.sequence_length > 1 else None)
        self.dataset_test = self.dataset_val

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)

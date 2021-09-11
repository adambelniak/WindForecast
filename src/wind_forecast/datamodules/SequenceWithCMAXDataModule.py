from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.datasets.CMAXDataset import CMAXDataset
from wind_forecast.datasets.ConcatDatasets import ConcatDatasets
from wind_forecast.datasets.SequenceDataset import SequenceDataset
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.utils import get_available_hdf_files_cmax, initialize_CMAX_list_IDs_and_synop_dates_for_sequence


class SequenceWithCMAXDataModule(LightningDataModule):

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

        self.target_param = config.experiment.target_parameter
        self.sequence_length = config.experiment.sequence_length
        self.synop_file = config.experiment.synop_file
        self.train_params = config.experiment.synop_train_features
        self.labels, self.label_mean, self.label_std = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]), dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=config.experiment.synop_from_year)

        target_param_index = [x[1] for x in self.train_params].index(self.target_param)
        print(self.label_mean[target_param_index])
        print(self.label_std[target_param_index])
        available_ids = get_available_hdf_files_cmax()
        self.cmax_IDs, self.dates = initialize_CMAX_list_IDs_and_synop_dates_for_sequence(available_ids, self.labels, self.target_param, self.sequence_length, config.experiment.future_sequence_length)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            dataset = ConcatDatasets(SequenceDataset(config=self.config, labels=self.labels, train=True), CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, train=True, normalize=True))
            length = len(dataset)
            self.dataset_train, self.dataset_val = random_split(dataset, [length - (int(length * self.val_split)),
                                                                          int(length * self.val_split)])
        elif stage == 'test':
            self.dataset_test = ConcatDatasets(SequenceDataset(config=self.config, labels=self.labels, train=False), CMAXDataset(config=self.config, train_IDs=self.cmax_IDs, train=False, normalize=True))

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
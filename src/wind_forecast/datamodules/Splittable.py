import json
import os
import sys

if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle
from typing import Optional

from pytorch_lightning import LightningDataModule

from wind_forecast.config.register import Config
from wind_forecast.consts import PREPARED_DATASETS_DIRECTORY
from wind_forecast.datamodules.DataModulesCache import DataModulesCache
from wind_forecast.util.common_util import split_dataset


class Splittable(LightningDataModule):

    def __init__(self, config: Config):
        super().__init__()
        self.val_split = config.experiment.val_split
        self.test_split = config.experiment.test_split
        self.dataset_split_mode = config.experiment.dataset_split_mode
        self.dataset_train = ...
        self.dataset_val = ...
        self.dataset_test = ...
        self.initialized = False

    def get_dataset_name(self, config: Config, stage: Optional[str] = None):
        exp = config.experiment
        name = f"{exp.synop_file}" \
               f"_{exp.synop_from_year}" \
               f"_{exp.synop_to_year}" \
               f"_{exp.target_parameter}" \
               f"_{str(exp.sequence_length)}" \
               f"_{'-'.join([f[1] for f in exp.synop_train_features])}" \
               f"_{str(exp.future_sequence_length)}" \
               f"_{'-'.join([f.column[1] for f in exp.synop_periodic_features])}" \
               f"_{str(exp.dataset_split_mode)}" \
               f"_{str(exp.prediction_offset)}" \
               f"_{os.path.basename(exp.train_parameters_config_file)}" \
               f"_{str(exp.test_split)}" \
               f"_{str(exp.val_split)}" \
               f"_{str(exp.normalization_type.value)}" \
               f"_{str(exp.use_gfs_data)}" \
               f"_{str(exp.use_all_gfs_params)}" \
               f"_{str(exp.cmax_from_year)}" \
               f"_{str(exp.cmax_to_year)}" \
               f"_{str(exp.cmax_normalization_type.value)}" \
               f"_{str(exp.cmax_sample_size)}" \
               f"_{str(exp.cmax_scaling_factor)}" \
               f"_{str(os.getenv('RUN_MODE'))}"

        return name + f"_{stage}"

    def split_dataset(self, config: Config, dataset, sequence_length):
        datasets = split_dataset(dataset,
                                 self.val_split,
                                 self.test_split,
                                 split_mode=self.dataset_split_mode,
                                 sequence_length=sequence_length if sequence_length > 1 else None)
        if self.val_split > 0:
            self.dataset_train, self.dataset_val, self.dataset_test = datasets
        else:
            self.dataset_train, self.dataset_test = datasets
            self.dataset_val = None

        print('Dataset train len: ' + str(len(self.dataset_train)))
        print('Dataset val len: ' + ('0' if self.dataset_val is None else str(len(self.dataset_val))))
        print('Dataset test len: ' + str(len(self.dataset_test)))

        train_dataset_name = self.get_dataset_name(config, 'fit')
        train_dataset_name_hash = str(abs(hash(train_dataset_name)) % (10 ** 8))
        val_dataset_name = self.get_dataset_name(config, 'validate')
        val_dataset_name_hash = str(abs(hash(val_dataset_name)) % (10 ** 8))
        test_dataset_name = self.get_dataset_name(config, 'test')
        test_dataset_name_hash = str(abs(hash(test_dataset_name)) % (10 ** 8))

        os.makedirs(PREPARED_DATASETS_DIRECTORY, exist_ok=True)

        with open(os.path.join(PREPARED_DATASETS_DIRECTORY, f"{train_dataset_name_hash}.pkl"), 'wb') as f:
            pickle.dump(self.dataset_train, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(PREPARED_DATASETS_DIRECTORY, f"{test_dataset_name_hash}.pkl"), 'wb') as f:
            pickle.dump(self.dataset_test, f, pickle.HIGHEST_PROTOCOL)
        DataModulesCache().cache_dataset('fit', self.dataset_train)
        DataModulesCache().cache_dataset('test', self.dataset_test)
        if self.dataset_val is not None:
            with open(os.path.join(PREPARED_DATASETS_DIRECTORY, f"{val_dataset_name_hash}.pkl"), 'wb') as f:
                pickle.dump(self.dataset_val, f, pickle.HIGHEST_PROTOCOL)

            DataModulesCache().cache_dataset('val', self.dataset_val)
        datasets_meta_path = os.path.join(PREPARED_DATASETS_DIRECTORY, "datasets_meta.json")
        if os.path.exists(datasets_meta_path):
            with open(datasets_meta_path) as f:
                dataset_hash_meta = json.load(f)
        else:
            dataset_hash_meta = {}
        dataset_hash_meta[train_dataset_name] = train_dataset_name_hash
        dataset_hash_meta[test_dataset_name] = test_dataset_name_hash
        dataset_hash_meta[val_dataset_name] = val_dataset_name_hash
        json.dump(dataset_hash_meta, open(datasets_meta_path, "w"))

    def load_from_disk(self, config: Config):
        datasets_meta_path = os.path.join(PREPARED_DATASETS_DIRECTORY, "datasets_meta.json")
        if os.path.exists(datasets_meta_path):
            with open(datasets_meta_path) as f:
                dataset_hash_meta = json.load(f)
            train_dataset_name = self.get_dataset_name(config, 'fit')
            if train_dataset_name not in dataset_hash_meta:
                return
            train_pickled_dataset = f"{os.path.join(PREPARED_DATASETS_DIRECTORY, dataset_hash_meta[train_dataset_name])}.pkl"

            test_dataset_name = self.get_dataset_name(config, 'test')
            if test_dataset_name not in dataset_hash_meta:
                return
            test_pickled_dataset = f"{os.path.join(PREPARED_DATASETS_DIRECTORY, dataset_hash_meta[test_dataset_name])}.pkl"

            if self.val_split > 0:
                val_dataset_name = self.get_dataset_name(config, 'validate')
                if val_dataset_name not in dataset_hash_meta:
                    return
                val_pickled_dataset = f"{os.path.join(PREPARED_DATASETS_DIRECTORY, dataset_hash_meta[val_dataset_name])}.pkl"

            if not os.path.exists(train_pickled_dataset) or not os.path.exists(test_pickled_dataset) or (
                    self.val_split > 0 and not os.path.exists(val_pickled_dataset)):
                return

            with open(train_pickled_dataset, 'rb') as f:
                self.dataset_train = pickle.load(f)
            with open(test_pickled_dataset, 'rb') as f:
                self.dataset_test = pickle.load(f)
            if self.val_split > 0:
                with open(val_pickled_dataset, 'rb') as f:
                    self.dataset_val = pickle.load(f)

            self.initialized = True

    def get_from_cache(self, stage: Optional[str] = None):
        if stage == 'test':
            cached_dataset = DataModulesCache().get_cached_dataset('test')
            if cached_dataset is not None:
                self.dataset_test = cached_dataset
                return True

        if stage == 'validate':
            cached_dataset = DataModulesCache().get_cached_dataset('val')
            if cached_dataset is not None:
                self.dataset_test = cached_dataset
                return True

        if stage == 'fit':
            cached_dataset = DataModulesCache().get_cached_dataset('fit')
            if cached_dataset is not None:
                self.dataset_test = cached_dataset
                return True

        return False

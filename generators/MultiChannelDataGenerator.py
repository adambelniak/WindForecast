import math
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import keras
from tqdm import tqdm

sys.path.insert(1, '../..')
sys.path.insert(2, '../../..')

from preprocess.synop.synop_preprocess import prepare_synop_dataset
from util.utils import GFS_DATASET_DIR, utc_to_local

NETCDF_FILE_REGEX = r"((\d+)-(\d+)-(\d+))-(\d+)-f(\d+).npy"
SYNOP_FILE = "KOZIENICE_488_data.csv"


class MultiChannelDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, parameters: [(str, str)], target_param: (str, str), synop_file=SYNOP_FILE,
                 batch_size=16, dim=(32, 32), shuffle=True, normalize=False):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.parameters = parameters
        self.target_param = target_param
        self.shuffle = shuffle
        self.labels = prepare_synop_dataset(synop_file, [target_param])
        self.on_epoch_end()
        self.mean, self.std = [], []

        if normalize:
            self.initialize_mean_and_std()

    def initialize_mean_and_std(self):
        for param in tqdm(self.parameters):
            sum, sqr_sum = 0, 0
            for id in tqdm(self.list_IDs):
                values = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], id))
                sum += np.sum(values)
                sqr_sum += pow(sum, 2)

            mean = sum / (len(self.list_IDs) * self.dim[0] * self.dim[1])
            self.mean.append(mean)
            self.std.append(math.sqrt(sqr_sum / (len(self.list_IDs) * self.dim[0] * self.dim[1]) - pow(mean, 2)))


    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        x = np.empty((self.batch_size, len(self.parameters), *self.dim))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            for j, param in enumerate(self.parameters):
                # Store sample
                x[i, j, ] = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], ID))
                x[i, j, ] = (x[i, j, ] - self.mean[j]) / self.std[j]

            # Store class
            date_matcher = re.match(NETCDF_FILE_REGEX, ID)

            date_from_filename = date_matcher.group(1)
            year = int(date_from_filename[:4])
            month = int(date_from_filename[5:7])
            day = int(date_from_filename[8:10])
            run = int(date_matcher.group(2))
            offset = int(date_matcher.group(3))
            forecast_date = utc_to_local(datetime(year, month, day) + timedelta(hours=run + offset))
            label = self.labels[self.labels["date"] == forecast_date][self.target_param]
            if len(label) == 0:
                print(forecast_date)
            y[i] = self.labels[self.labels["date"] == forecast_date][self.target_param].values[0]

        x = np.einsum('klij->kijl', x)
        return x, y
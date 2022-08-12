import os
import re
from datetime import datetime

import h5py
import math

import numpy as np
from skimage.measure import block_reduce

from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset
from wind_forecast.util.cmax_util import CMAX_DATASET_DIR, CMAX_MIN, CMAX_MAX

"""
Used for pre-traning convolutional autoencoder.
Dedicated for 10-minutes radar images.
"""
class CMAXCAEDataset(BaseDataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, IDs):
        super().__init__()
        self.config = config
        self.dim = config.experiment.cmax_sample_size
        self.normalization_type = config.experiment.cmax_normalization_type
        self.data = IDs

        self.cmax_values = {}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        id = self.data[index]

        return self.__data_generation(id)

    def __data_generation(self, id: str):
        # Initialization
        x = np.empty((math.ceil(self.dim[0] / self.config.experiment.cmax_scaling_factor),
                      math.ceil(self.dim[1] / self.config.experiment.cmax_scaling_factor)))

        # Generate data
        data = self.get_cmax_array_from_file(id)
        if data is None:
            x[:, ] = np.zeros_like(x)
        else:
            x[:, ] = data
            x[:, ] = (x[:, ] - CMAX_MIN) / (CMAX_MAX - CMAX_MIN)
        return x

    def date_from_h5y_file(self, filename: str):
        date_matcher = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})0000dBZ\.cmax\.h5", filename)

        year = int(date_matcher.group(1))
        month = int(date_matcher.group(2))
        day = int(date_matcher.group(3))
        hour = int(date_matcher.group(4))
        minutes = int(date_matcher.group(5))
        date = datetime(year, month, day, hour, minutes)
        return date

    def get_cmax_array_from_file(self, filename: str):
        def filter(b, axis):
            x1 = (b > 75).sum(axis=axis)
            ret = np.max(b, axis=axis)
            ret[np.where(x1 <= 4)] = 0
            return ret
        try:
            with h5py.File(os.path.join(CMAX_DATASET_DIR, filename), 'r') as hdf:
                data = np.array(hdf.get('dataset1').get('data1').get('data'))
                mask = np.where((data >= 255) | (data <= 0))
                data[mask] = 0
                data[data == None] = 0
                resampled = block_reduce(data, block_size=(4, 4), func=filter).squeeze()
                return resampled
        except Exception:
            return None

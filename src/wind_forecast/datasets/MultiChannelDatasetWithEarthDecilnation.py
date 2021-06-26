import math
import os
import re
from datetime import datetime, timedelta

import torch
from tqdm import tqdm
import numpy as np

from wind_forecast.consts import NETCDF_FILE_REGEX
from wind_forecast.util.utils import GFS_DATASET_DIR, utc_to_local, declination_of_earth


class MultiChannelDatasetWithEarthDecilnation(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, train=True, normalize=True):
        'Initialization'
        self.list_IDs = list_IDs
        length = len(self.list_IDs)
        training_data, validation_data = self.list_IDs[:int(length * 0.8)], self.list_IDs[int(length * 0.8):]
        if train:
            data = training_data
        else:
            data = validation_data

        self.data = data

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
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.data[index]

        x1, x2, y = self.__data_generation(ID)

        return [x1, x2], y

    def __data_generation(self, ID):
        # Initialization
        x1 = np.empty((len(self.parameters), *self.dim))

        # Generate data
        for j, param in enumerate(self.parameters):
            # Store sample
            x1[j, ] = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], ID))
            x1[j, ] = (x1[j, ] - self.mean[j]) / self.std[j]

        # Store class
        date_matcher = re.match(NETCDF_FILE_REGEX, ID)

        date_from_filename = date_matcher.group(1)
        year = int(date_from_filename[:4])
        month = int(date_from_filename[5:7])
        day = int(date_from_filename[8:10])
        run = int(date_matcher.group(2))
        offset = int(date_matcher.group(3))
        forecast_date = utc_to_local(datetime(year, month, day) + timedelta(hours=run + offset))
        x2 = declination_of_earth(forecast_date) / 23.45
        label = self.labels[self.labels["date"] == forecast_date][self.target_param]
        y = label.values[0]

        x1 = np.einsum('lij->ijl', x1)
        return x1, x2, y
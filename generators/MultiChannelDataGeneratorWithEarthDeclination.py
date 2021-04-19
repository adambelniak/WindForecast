import os
import re
from datetime import datetime, timedelta

from generators.MultiChannelDataGenerator import MultiChannelDataGenerator, SYNOP_FILE, NETCDF_FILE_REGEX
import numpy as np

from util.utils import GFS_DATASET_DIR, utc_to_local, declination_of_earth


class MultiChannelDataGeneratorWithEarthDeclination(MultiChannelDataGenerator):
    def __init__(self, list_IDs, parameters: [(str, str)], target_param: (str, str), synop_file=SYNOP_FILE,
                 batch_size=16, dim=(32, 32), shuffle=True, normalize=False):
        super().__init__(list_IDs, parameters, target_param, synop_file, batch_size, dim, shuffle, normalize)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x1, x2, y = self.__data_generation(list_IDs_temp)

        return [x1, x2], y

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        x1 = np.empty((self.batch_size, len(self.parameters), *self.dim))
        x2 = np.empty(self.batch_size, dtype=float)
        y = np.empty(self.batch_size, dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            for j, param in enumerate(self.parameters):
                # Store sample
                x1[i, j, ] = np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], ID))
                x1[i, j, ] = (x1[i, j, ] - self.mean[j]) / self.std[j]

            # Store class
            date_matcher = re.match(NETCDF_FILE_REGEX, ID)

            date_from_filename = date_matcher.group(1)
            year = int(date_from_filename[:4])
            month = int(date_from_filename[5:7])
            day = int(date_from_filename[8:10])
            run = int(date_matcher.group(2))
            offset = int(date_matcher.group(3))
            forecast_date = utc_to_local(datetime(year, month, day) + timedelta(hours=run + offset))
            x2[i] = declination_of_earth(forecast_date) / 23.45
            label = self.labels[self.labels["date"] == forecast_date][self.target_param]
            if len(label) == 0:
                print(forecast_date)
            y[i] = self.labels[self.labels["date"] == forecast_date][self.target_param].values[0]

        x1 = np.einsum('klij->kijl', x1)
        return x1, x2, y
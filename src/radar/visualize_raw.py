import os
from datetime import datetime

import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

from radar.fetch_radar_CMAX import CMAX_DATASET_DIR
from util.util import prep_zeros_if_needed
from wind_forecast.consts import CMAX_H5_FILENAME_FORMAT

if __name__ == "__main__":
    date = datetime(2017, 8, 11, 20)
    filename = CMAX_H5_FILENAME_FORMAT.format(str(date.year), prep_zeros_if_needed(str(date.month), 1),
                                              prep_zeros_if_needed(str(date.day), 1),
                                              prep_zeros_if_needed(str(date.hour), 1),
                                              '00')
    try:
        with h5py.File(os.path.join(CMAX_DATASET_DIR, filename), 'r') as hdf:
            data = np.array(hdf.get('dataset1').get('data1').get('data'))
            mask = np.where((data >= 255) | (data <= 0))
            data[mask] = 0
            data[data == None] = 0
            def filter(b, axis):
                x1 = (b > 75).sum(axis=axis)
                ret = np.max(b, axis=axis)
                ret[np.where(x1 <= 4)] = 0
                return ret
            resampled = block_reduce(data, block_size=(4, 4),
                                     func=filter).squeeze()
            resampled = np.uint8(resampled)
            sns.heatmap(resampled)
            plt.show()
    except OSError:
        pass

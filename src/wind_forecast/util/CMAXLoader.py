import os
import sys
if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle
from datetime import datetime
import numpy as np

CMAX_DATASET_DIR = os.environ.get('CMAX_DATASET_DIR')
CMAX_DATASET_DIR = 'data' if CMAX_DATASET_DIR is None else CMAX_DATASET_DIR


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CMAXLoader(metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.cmax_images = {}
        self.pickle_dir = os.path.join(CMAX_DATASET_DIR, 'pkl')

    @staticmethod
    def get_date_key(date):
        return datetime.strftime(date, "%Y%m%d%H")

    def get_cmax_image(self, date_key):
        pickle_filename = date_key[:-4]

        if date_key not in self.cmax_images:
            with open(os.path.join(self.pickle_dir, pickle_filename + ".pkl"), 'rb') as f:
                self.cmax_images.update(pickle.load(f))

        return np.array(self.cmax_images[date_key])

    def get_all_loaded_cmax_images(self):
        return self.cmax_images




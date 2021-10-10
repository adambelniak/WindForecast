import os
import sys

from wind_forecast.loaders.Singleton import Singleton

if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle
from datetime import datetime
import numpy as np
from gfs_archive_0_25.utils import prep_zeros_if_needed


GFS_DATASET_DIR = os.environ.get('GFS_DATASET_DIR')
GFS_DATASET_DIR = 'gfs_data' if GFS_DATASET_DIR is None else GFS_DATASET_DIR


class GFSLoader(metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.gfs_images = {}
        self.pickle_dir = os.path.join(GFS_DATASET_DIR, 'pkl')

    @staticmethod
    def get_date_key(date):
        return datetime.strftime(date, "%Y%m%d%H")

    def get_gfs_image(self, date_key: str, gfs_parameter, from_offset: int):
        param_key = f"{gfs_parameter['name']}_{gfs_parameter['level']}"
        pickle_filename = f"{gfs_parameter['name']}_{gfs_parameter['level']}_{prep_zeros_if_needed(str(from_offset), 2)}"

        if param_key not in self.gfs_images or str(from_offset) not in self.gfs_images[param_key]:
            if param_key not in self.gfs_images:
                self.gfs_images[param_key] = {}
            with open(os.path.join(self.pickle_dir, pickle_filename + ".pkl"), 'rb') as f:
                self.gfs_images[param_key][str(from_offset)] = pickle.load(f)

        return np.array(self.gfs_images[param_key][str(from_offset)][date_key]) if date_key in \
                                                                                   self.gfs_images[param_key][str(from_offset)] else None

    def get_all_loaded_gfs_images(self):
        return self.gfs_images




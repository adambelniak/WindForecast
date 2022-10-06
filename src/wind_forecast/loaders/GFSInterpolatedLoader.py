import os
import sys
from datetime import datetime
from typing import Dict, Union

from gfs_archive_0_25.gfs_processor.Coords import Coords
from gfs_archive_0_25.utils import get_nearest_coords
from gfs_common.common import GFS_SPACE
from wind_forecast.loaders.Singleton import Singleton
from wind_forecast.loaders.GFSLoader import GFSLoader
import numpy as np
from gfs_archive_0_25.utils import prep_zeros_if_needed
if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle
GFS_DATASET_DIR = os.environ.get('GFS_DATASET_DIR')
GFS_DATASET_DIR = 'gfs_data' if GFS_DATASET_DIR is None else GFS_DATASET_DIR


def get_param_key(param):
    return f"{param['name']}_{param['level']}"


class Interpolator(metaclass=Singleton):
    def __init__(self, target_point: Coords) -> None:
        super().__init__()
        self.target_point = target_point
        self.nearest_coords = get_nearest_coords(target_point)
        self.lat_index = int((GFS_SPACE.nlat - self.nearest_coords.slat) * 4)
        self.lon_index = int((GFS_SPACE.elon - self.nearest_coords.wlon) * 4)
        self.y_factor = (self.nearest_coords.nlat - target_point.nlat) / (
                self.nearest_coords.nlat - self.nearest_coords.slat)
        self.x_factor = (self.nearest_coords.elon - target_point.elon) / (
                self.nearest_coords.elon - self.nearest_coords.wlon)

    def __call__(self, gfs_data: np.ndarray) -> float:
        f_x_y1 = self.x_factor * gfs_data[self.lat_index + 1, self.lon_index] + (1 - self.x_factor) * gfs_data[
            self.lat_index + 1, self.lon_index + 1]
        f_x_y2 = self.x_factor * gfs_data[self.lat_index, self.lon_index] + (1 - self.x_factor) * gfs_data[
            self.lat_index, self.lon_index + 1]
        f_x_y = self.y_factor * f_x_y1 + (1 - self.y_factor) * f_x_y2
        return f_x_y


"""
Interpolates values in a space dimension
"""
class GFSInterpolatedLoader:
    def __init__(self, target_point: Coords) -> None:
        super().__init__()
        self.gfs_interpolated_values = {}
        self.pickle_dir = os.path.join(GFS_DATASET_DIR, 'pkl')
        self.interpolator = Interpolator(target_point)

    def get_gfs_value_for_date(self, date: datetime, gfs_parameter: Dict, from_offset: int) -> Union[np.ndarray, None]:
        date_key = GFSLoader.get_date_key(date)
        param_key = get_param_key(param=gfs_parameter)
        if param_key not in self.gfs_interpolated_values \
                or str(from_offset) not in self.gfs_interpolated_values[param_key]\
                or date_key not in self.gfs_interpolated_values[param_key][str(from_offset)]:

            values_for_offset = self.get_gfs_images_for_offset(gfs_parameter, from_offset)
            if values_for_offset is None:
                return None

            for item in values_for_offset.items():
                item_date_key = item[0]
                value = self.interpolator(item[1])
                if param_key not in self.gfs_interpolated_values:
                    self.gfs_interpolated_values[param_key] = {}
                if str(from_offset) not in self.gfs_interpolated_values[param_key]:
                    self.gfs_interpolated_values[param_key][str(from_offset)] = {}
                self.gfs_interpolated_values[param_key][str(from_offset)][item_date_key] = value

        return np.array(self.gfs_interpolated_values[param_key][str(from_offset)][date_key])\
            if date_key in self.gfs_interpolated_values[param_key][str(from_offset)] else None

    def get_gfs_images_for_offset(self, gfs_parameter: Dict, from_offset: int):
        param_key = f"{gfs_parameter['name']}_{gfs_parameter['level']}"
        pickle_filename = f"{param_key}_{prep_zeros_if_needed(str(from_offset), 2)}"

        with open(os.path.join(self.pickle_dir, pickle_filename + ".pkl"), 'rb') as f:
            return pickle.load(f)

    def get_all_loaded_gfs_values(self):
        return self.gfs_interpolated_values

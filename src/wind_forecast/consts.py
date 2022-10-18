import os
from enum import Enum
from pathlib import Path

CREATED_AT_COLUMN_NAME = "created_at"
NETCDF_FILE_REGEX = r"((\d+)-(\d+)-(\d+))-(\d+)-f(\d+).npy"
CMAX_NPY_FILE_REGEX = r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})0000dBZ\.cmax\.h5\.npy"
CMAX_H5_FILE_REGEX = r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})0000dBZ\.cmax\.h5"
DATE_KEY_REGEX = r"(\d{4})(\d{2})(\d{2})(\d{2})"
CMAX_NPY_FILENAME_FORMAT = "{0}{1}{2}{3}{4}0000dBZ.cmax.h5.pkl"
CMAX_H5_FILENAME_FORMAT = "{0}{1}{2}{3}{4}0000dBZ.cmax.h5"
SYNOP_DATASETS_DIRECTORY = os.path.join(Path(__file__).parent, "..", "data", "synop")
PREPARED_DATASETS_DIRECTORY = os.path.join(Path(__file__).parent, "..", "datasets")

STATION_META = {
    'Warsaw': {
        'lat': 52.162523,
        'lon': 20.961864,
        'synop_file': "WARSZAWA-OKECIE_375_data.csv"
    },
    'Hel': {
        'lat': 54.603577,
        'lon': 18.811969,
        'synop_file': "HEL_135_data.csv"
    },
    'Kozienice': {
        'lat': 51.564790,
        'lon': 21.543629,
        'synop_file': "KOZIENICE_488_data.csv"
    }
}


class BatchKeys(Enum):
    SYNOP_PAST_X = 'synop_past_x'
    SYNOP_PAST_Y = 'synop_past_y'
    SYNOP_FUTURE_Y = 'synop_future_y'
    SYNOP_FUTURE_X = 'synop_future_x'
    GFS_PAST_X = 'gfs_past_x'
    GFS_PAST_Y = 'gfs_past_y'
    GFS_FUTURE_Y = 'gfs_future_y'
    GFS_FUTURE_X = 'gfs_future_x'
    GFS_SYNOP_PAST_DIFF = 'GFS_SYNOP_PAST_DIFF'
    GFS_SYNOP_FUTURE_DIFF = 'GFS_SYNOP_FUTURE_DIFF'
    DATES_PAST = 'dates_past'
    DATES_FUTURE = 'dates_future'
    DATES_TENSORS = 'dates_embedding'
    CMAX_PAST = 'cmax_past'
    CMAX_FUTURE = 'cmax_future'
    PREDICTIONS = 'predictions'

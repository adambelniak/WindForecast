import os
from enum import Enum
from pathlib import Path

CREATED_AT_COLUMN_NAME = "created_at"
NETCDF_FILE_REGEX = r"((\d+)-(\d+)-(\d+))-(\d+)-f(\d+).npy"
CMAX_NPY_FILE_REGEX = r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})0000dBZ\.cmax\.h5\.npy"
DATE_KEY_REGEX = r"(\d{4})(\d{2})(\d{2})(\d{2})"
CMAX_NPY_FILENAME_FORMAT = "{0}{1}{2}{3}{4}0000dBZ.cmax.h5.pkl"
SYNOP_DATASETS_DIRECTORY = os.path.join(Path(__file__).parent, "..", "data", "synop")

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
    SYNOP_INPUTS = 'synop_inputs'
    SYNOP_TARGETS = 'synop_targets'
    ALL_SYNOP_TARGETS = 'all_synop_targets'
    GFS_INPUTS = 'gfs_inputs'
    GFS_TARGETS = 'gfs_targets'
    ALL_GFS_TARGETS = 'all_gfs_targets'
    DATES_INPUTS = 'dates_inputs'
    DATES_TARGETS = 'dates_targets'
    DATES_EMBEDDING = 'dates_embedding'
    CMAX_INPUTS = 'cmax_inputs'
    CMAX_TARGETS = 'cmax_targets'

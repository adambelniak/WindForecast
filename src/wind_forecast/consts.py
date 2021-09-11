import os
from pathlib import Path

CREATED_AT_COLUMN_NAME = "created_at"
NETCDF_FILE_REGEX = r"((\d+)-(\d+)-(\d+))-(\d+)-f(\d+).npy"
CMAX_FILENAME_FORMAT = "{0}{1}{2}{3}{4}0000dBz.cmax.h5"
SYNOP_DATASETS_DIRECTORY = os.path.join(Path(__file__).parent, "..", "data", "synop")

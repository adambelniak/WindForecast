import os
from pathlib import Path

CREATED_AT_COLUMN_NAME = "created_at"
NETCDF_FILE_REGEX = r"((\d+)-(\d+)-(\d+))-(\d+)-f(\d+).npy"
SYNOP_DATASETS_DIRECTORY = os.path.join(Path(__file__).parent, "..", "data", "synop")

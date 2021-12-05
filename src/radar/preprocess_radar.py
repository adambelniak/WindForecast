import os
import re
from datetime import datetime

import h5py
from tqdm import tqdm
import numpy as np
from radar.fetch_radar_CMAX import CMAX_DATASET_DIR
from skimage.measure import block_reduce
import sys
if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle


def date_from_h5y_file(filename: str):
    date_matcher = re.match(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})0000dBZ\.cmax\.h5", filename)

    year = int(date_matcher.group(1))
    month = int(date_matcher.group(2))
    day = int(date_matcher.group(3))
    hour = int(date_matcher.group(4))
    minutes = int(date_matcher.group(5))
    date = datetime(year, month, day, hour, minutes)
    return date


if __name__ == "__main__":
    cmax_pkl_dir = os.path.join(CMAX_DATASET_DIR, 'pkl')
    os.makedirs(cmax_pkl_dir, exist_ok=True)

    matcher = re.compile(r"(\d{4})(\d{2})\d{4}000000dBZ\.cmax\.h5")
    print(f"Scanning {CMAX_DATASET_DIR} looking for CMAX files.")
    hour_files = [f.name for f in tqdm(os.scandir(CMAX_DATASET_DIR)) if matcher.match(f.name)]
    hour_files.sort()

    month = 1
    year = 2017
    date = datetime(year, month, 1, 0, 0)
    while date < datetime(2021, 9, 1):
        cmax_dict = {}
        dict_name = datetime.strftime(date, "%Y%m")
        if not os.path.exists(os.path.join(cmax_pkl_dir, f"{dict_name}_meta.pkl")):
            files = [file for file in hour_files if int(matcher.match(file).group(1)) == year
                     and int(matcher.match(file).group(2)) == month]
            if len(files) > 0:
                for file in tqdm(files):
                    with h5py.File(os.path.join(CMAX_DATASET_DIR, file), 'r') as hdf:
                        date = date_from_h5y_file(file)
                        date_key = datetime.strftime(date, "%Y%m%d%H")
                        data = np.array(hdf.get('dataset1').get('data1').get('data'))
                        mask = np.where(data >= 255)
                        data[mask] = 0
                        resampled = block_reduce(data, block_size=(4, 4), func=np.max)
                        resampled = np.uint8(resampled)
                        cmax_dict[date_key] = resampled

                with open(os.path.join(cmax_pkl_dir, f"{dict_name}.pkl"), 'wb') as f:
                    pickle.dump(cmax_dict, f, pickle.HIGHEST_PROTOCOL)

                with open(os.path.join(cmax_pkl_dir, f"{dict_name}_meta.pkl"), 'wb') as f:
                    pickle.dump(list(cmax_dict.keys()), f, pickle.HIGHEST_PROTOCOL)

        month = month + 1 if month < 12 else 1
        if month == 1:
            year += 1
        date = datetime(year, month, 1, 0, 0)

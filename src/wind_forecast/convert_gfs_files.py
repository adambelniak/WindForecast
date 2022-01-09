import os
import pickle

from tqdm import tqdm

from gfs_archive_0_25.utils import prep_zeros_if_needed

# from gfs_common.common import GFS_PARAMETERS
import re
import numpy as np

from wind_forecast.loaders.GFSLoader import GFSLoader
from wind_forecast.util.gfs_util import GFS_DATASET_DIR, date_from_gfs_np_file

GFS_PARAMETERS = [
{
    "name": "HGT",
    "level": "ISBL_500"
}
]

"""
Converts GFS forecast kept in numpy .npy files to pickle .pkl files, grouped by offset and parameter.
It's a convenient form of keeping and loading data and GFSLoader works on those files. 
"""
if __name__ == "__main__":
    pkl_dir = os.path.join(GFS_DATASET_DIR, 'pkl')
    for offset in tqdm(range(24, 36, 3)):
        for param in GFS_PARAMETERS:
            meta_keys = []
            gfs_images = {}
            files_matcher = re.compile(rf".*-f{prep_zeros_if_needed(str(offset), 2)}\.npy")
            for file in tqdm([file for file in tqdm(os.scandir(os.path.join(GFS_DATASET_DIR, param['name'], param['level']))) if files_matcher.match(file.name)]):
                date = date_from_gfs_np_file(file.name)
                date_key = GFSLoader.get_date_key(date)
                meta_keys.append(date_key)
                values = np.load(file)
                if len(values.shape) == 3:
                    # shape of values changed on RDA :/
                    values = values[0]
                gfs_images[date_key] = values

            with open(os.path.join(pkl_dir, f"{param['name']}_{param['level']}_{prep_zeros_if_needed(str(offset), 2)}_meta.pkl"), 'wb') as f:
               pickle.dump(meta_keys, f, pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(pkl_dir, f"{param['name']}_{param['level']}_{prep_zeros_if_needed(str(offset), 2)}.pkl"), 'wb') as f:
               pickle.dump(gfs_images, f, pickle.HIGHEST_PROTOCOL)
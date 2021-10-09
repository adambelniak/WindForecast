import pickle
from datetime import datetime, timedelta
import os
import re
import sys
# import time
from pathlib import Path
from zipfile import ZipFile, BadZipFile
# import numpy as np
# from shutil import copyfile

# import h5py
import requests
# from skimage.measure import block_reduce
# from sklearn.utils import resample
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tqdm import tqdm

from gfs_archive_0_25.utils import prep_zeros_if_needed
# from wind_forecast.util.cmax_util import get_available_hdf_files_cmax_hours, get_cmax_npy, date_from_cmax_npy_file

DATA_URL = "https://danepubliczne.imgw.pl/datastore/getfiledown/Arch/Polrad/Produkty/POLCOMP/COMPO_CMAX_250.comp.cmax"
CMAX_DATASET_DIR = os.environ['CMAX_DATASET_DIR']
CMAX_DATASET_DIR = 'data' if CMAX_DATASET_DIR is None else CMAX_DATASET_DIR


def check_file_status(filepath):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    downloaded = size / (1024 * 1024)
    sys.stdout.write('%.3f MB Downloaded' % downloaded)
    sys.stdout.flush()


def get_zip(date: datetime):
    output_dir = CMAX_DATASET_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"COMPO_CMAX_250.comp.cmax_{date.strftime('%Y-%m-%d')}.zip"
    url = f"{DATA_URL}/{str(date.year)}/{prep_zeros_if_needed(str(date.month), 1)}/{filename}"
    req = requests.get(url, stream=True)
    with open(os.path.join(output_dir, filename), 'wb') as outfile:
        chunk_size = 1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            check_file_status(os.path.join(output_dir, filename))


def extract_zip(date: datetime):
    dir_with_zip = CMAX_DATASET_DIR
    filename = f"COMPO_CMAX_250.comp.cmax_{date.strftime('%Y-%m-%d')}.zip"
    tmp_dir = os.path.join(dir_with_zip, 'tmp')
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(os.path.join(dir_with_zip, filename), 'r') as zip:
            zip.extractall(path=tmp_dir)
        os.remove(os.path.join(dir_with_zip, filename))
        for file in os.listdir(tmp_dir):
            if not file.endswith(".h5"):
                os.remove(os.path.join(tmp_dir, file))
            else:
                os.rename(os.path.join(tmp_dir, file), os.path.join(dir_with_zip, file))
    except BadZipFile:
        return


def get_all_zips():
    date = datetime(2021, 6, 19)

    while date != datetime(2021, 9, 5):
        print(f"Fetching zip for date {date.strftime('%Y-%m-%d')}")
        get_zip(date)
        extract_zip(date)
        date = date + timedelta(days=1)


if __name__ == "__main__":
    get_all_zips()

    #
    # npy_dir = os.path.join(CMAX_DATASET_DIR, 'npy')
    # os.makedirs(npy_dir, exist_ok=True)
    # matcher = re.compile(r"(\d{4})\d{6}000000dBZ\.cmax\.h5")
    # files = set([f.name for f in tqdm(os.scandir(os.path.join(CMAX_DATASET_DIR))) if matcher.match(f.name)])
    #
    # for file in tqdm(files):
    #     if not os.path.exists(os.path.join(npy_dir, file + '.npy')):
    #         with h5py.File(os.path.join(CMAX_DATASET_DIR, file), 'r') as hdf:
    #             data = np.array(hdf.get('dataset1').get('data1').get('data'))
    #             mask = np.where((data >= 255) | (data < 0))
    #             data[mask] = 0
    #             values = block_reduce(data, block_size=(4, 4), func=np.mean)
    #             np.save(os.path.join(npy_dir, file + '.npy'), np.int8(values), allow_pickle=False)

    # obj = get_cmax('2021010100000000dBZ.cmax.h5.npy')
    # print(obj)
    # os.makedirs(pkl_dir, exist_ok=True)
    #
    # hour_files = get_available_hdf_files_cmax_hours()
    # month = 1
    # year = 2017
    # date = datetime(year, month, 1, 0, 0)
    # pkl_dir = os.path.join(CMAX_DATASET_DIR, 'pkl')
    # matcher = re.compile(r"(\d{4})(\d{2})\d{4}000000dBZ\.cmax\.h5\.npy")
    # while date < datetime(2021, 9, 1):
    #     cmax_dict = {}
    #     dict_name = datetime.strftime(date, "%Y%m")
    #     if not os.path.exists(os.path.join(pkl_dir, f"{dict_name}_meta.pkl")):
    #         files = [file for file in hour_files if int(matcher.match(file).group(1)) == year
    #                  and int(matcher.match(file).group(2)) == month]
    #         if len(files) > 0:
    #             for file in tqdm(files):
    #                 date = date_from_cmax_npy_file(file)
    #                 date_key = datetime.strftime(date, "%Y%m%d%H")
    #                 cmax_dict[date_key] = "" #get_cmax_npy(file)
    #
    #             # with open(os.path.join(pkl_dir, f"{dict_name}.pkl"), 'wb') as f:
    #             #     pickle.dump(cmax_dict, f, pickle.HIGHEST_PROTOCOL)
    #
    #             with open(os.path.join(pkl_dir, f"{dict_name}_meta.pkl"), 'wb') as f:
    #                 pickle.dump(list(cmax_dict.keys()), f, pickle.HIGHEST_PROTOCOL)
    #
    #     month = month + 1 if month < 12 else 1
    #     if month == 1:
    #         year += 1
    #     date = datetime(year, month, 1, 0, 0)

    # start = time.time()
    # for i in tqdm(range(0, 1000)):
    #     try:
    #         if not os.path.exists(os.path.join(CMAX_DATASET_DIR, 'hours', hour_files[i] + '.npy')):
    #             values = np.int8(get_hdf(hour_files[i], 1))
    #             np.save(os.path.join(CMAX_DATASET_DIR, 'hours', hour_files[i] + '.npy'), values)
    #     except OSError:
    #         pass
        # values = get_hdf(os.path.join(CMAX_DATASET_DIR, 'hours', hour_files[i]), 8)
    # end = time.time()
    # print(f"Processing of hdf took {end - start}")

    # start = time.time()
    # os.makedirs(os.path.join(CMAX_DATASET_DIR, 'npy'), exist_ok=True)
    # for file in tqdm(hour_files):
        # try:
        #     with h5py.File(os.path.join(CMAX_DATASET_DIR, file), 'r') as hdf:
        #         data = np.array(hdf.get('dataset1').get('data1').get('data'))
        #         mask = np.where((data == 255) | (data < 0))
        #         data[mask] = 0
        # np.load(os.path.join(CMAX_DATASET_DIR, 'npy', file))
        # np.save(os.path.join(CMAX_DATASET_DIR, 'npy', file), np.int8(values), allow_pickle=False)
                # np.save(os.path.join(CMAX_DATASET_DIR, 'npy', file + '.npy'), values, allow_pickle=False)
        # except OSError:
        #     pass
            # values = get_hdf(hour_files[i], 1)

            # end = time.time()
    # print(f"Processing of np took {end - start}")
    #
        # if not os.path.exists(os.path.join(CMAX_DATASET_DIR, 'hours', hour_files[i] + '.npy')):
        #     values = get_hdf(hour_files[i], 1)
        #     np.save(os.path.join(CMAX_DATASET_DIR, 'hours', hour_files[i] + '.npy'), values)



    # start = time.time()
    # for file in hour_files:
    #     values = get_hdf(file, 8)
    # end = time.time()
    #
    # print(f"Processing of hdf took {end - start}")
    #
    # start = time.time()
    # for file in hour_files:
    #     values = get_hdf(file, 8)
    # end = time.time()

    # with h5py.File(os.path.join(CMAX_DATASET_DIR, '2020083017100000dBZ.cmax.h5'), 'r') as hdf:
    #     data = np.array(hdf.get('dataset1').get('data1').get('data'))
    #     # mask = np.load(os.path.join(CMAX_DATASET_DIR, 'mask.npy'))
    #     # values = data - mask
    #     mask = np.where(data == 255)
    #     data[mask] = data[mask] - 255
    #     resampled = block_reduce(data, block_size=(1, 1), func=np.mean)
    #     sns.heatmap(resampled)
    #     plt.show()

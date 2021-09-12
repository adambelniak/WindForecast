import datetime
import os
import sys
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import numpy as np
import matplotlib.pyplot as plt

import h5py
import requests
import seaborn

from gfs_archive_0_25.utils import prep_zeros_if_needed

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


def get_zip(date: datetime.datetime):
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


def extract_zip(date: datetime.datetime):
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
    date = datetime.datetime(2020, 9, 7)

    while date != datetime.datetime(2021, 9, 5):
        print(f"Fetching zip for date {date.strftime('%Y-%m-%d')}")
        get_zip(date)
        extract_zip(date)
        date = date + datetime.timedelta(days=1)


if __name__ == "__main__":
    get_all_zips()





#h5_filepath = "D:\\WindForecast\\cmax\\2017\\2017010100000000dBZ.cmax.h5"
    # with h5py.File(h5_filepath, 'r') as hdf:
    #     data = np.array(hdf.get('dataset1').get('data1').get('data'))
    #     data[np.abs(data) < 255] = 0
    #     np.save("D:\\WindForecast\\cmax\\mask.npy", data)
    #     seaborn.heatmap(data)
    #     plt.show()

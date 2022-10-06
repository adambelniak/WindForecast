import argparse
import os
import re
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker
from tqdm import tqdm

from radar.fetch_radar_CMAX import CMAX_DATASET_DIR


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
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_year', help='Start processing from this year. Must be between 2011 and 2022', type=int, default=2016)
    parser.add_argument('--from_month', help='Start processing from this month', type=int, default=1)
    parser.add_argument('--to_year', help='Process up to this year', default=2021, type=int)
    parser.add_argument('--to_month', help='Process up to this month', default=12, type=int)

    args = parser.parse_args()
    assert 2011 < args.from_year < 2022, "2011 < args.from_year < 2022"
    assert 2011 < args.to_year < 2022, "2011 < args.to_year < 2022"
    assert args.from_year <= args.to_year, "args.from_year <= args.to_year"
    assert 0 < args.from_month < 13, "0 < args.from_month < 13"
    assert 0 < args.to_month < 13, "0 < args.to_month < 13"
    assert args.from_year < args.to_year or (args.from_year == args.to_year and args.from_month < args.to_month),\
        "args.from_year < args.to_year or (args.from_year == args.to_year and args.from_month < args.to_month)"

    cmax_pkl_dir = os.path.join(CMAX_DATASET_DIR, 'pkl')

    os.makedirs(cmax_pkl_dir, exist_ok=True)

    matcher = re.compile(r"(\d{4})(\d{2})\d{4}000000dBZ\.cmax\.h5")
    print(f"Scanning {CMAX_DATASET_DIR} looking for CMAX hourly files.")
    hour_files = [f.name for f in tqdm(os.scandir(CMAX_DATASET_DIR)) if matcher.match(f.name)]
    hour_files.sort()

    month = args.from_month
    year = args.from_year
    date = datetime(year, month, 1, 0, 0)
    cmax_hist = np.zeros((256))
    while date < datetime(args.to_year + (1 if args.to_month == 12 else 0), 1 if args.to_month == 12 else args.to_month + 1, 1):
        cmax_dict = {}
        dict_name = datetime.strftime(date, "%Y%m")
        files = [file for file in hour_files if int(matcher.match(file).group(1)) == year
                 and int(matcher.match(file).group(2)) == month]
        if len(files) > 0:
            for file in tqdm(files):
                try:
                    with h5py.File(os.path.join(CMAX_DATASET_DIR, file), 'r') as hdf:
                        date = date_from_h5y_file(file)
                        date_key = datetime.strftime(date, "%Y%m%d%H")
                        data = np.array(hdf.get('dataset1').get('data1').get('data'))
                        mask = np.where((data >= 255) | (data <= 0))
                        data[mask] = 0
                        data[data == None] = 0
                        data[mask] = 0
                        hist, bins = np.histogram(data.flatten(), bins=np.arange(257))
                        cmax_hist += hist
                except Exception:
                    print(f"Error with processing file {os.path.join(CMAX_DATASET_DIR, file)}")

        month = month + 1 if month < 12 else 1
        if month == 1:
            year += 1
        date = datetime(year, month, 1, 0, 0)


    ax = sns.barplot(y=cmax_hist[1:], x=np.arange(255), color='b') # do not show 0s
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.show()
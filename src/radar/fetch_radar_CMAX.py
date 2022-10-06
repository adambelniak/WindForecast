import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import requests

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
            if not file.endswith(".h5") or os.path.exists(os.path.join(dir_with_zip, file)):
                os.remove(os.path.join(tmp_dir, file))
            else:
                os.rename(os.path.join(tmp_dir, file), os.path.join(dir_with_zip, file))
    except BadZipFile:
        print(f"BadZipFile thrown for file {filename}")
        return


def get_all_zips():
    parser = argparse.ArgumentParser()

    parser.add_argument('--from_year', help='Start fetching from this year. Must be between 2011 and 2022', type=int, default=2016)
    parser.add_argument('--from_month', help='Start fetching from this month', type=int, default=1)
    parser.add_argument('--to_year', help='Fetch up to this year', default=2021, type=int)
    parser.add_argument('--to_month',
                        help='Fetch up to this month', default=12, type=int)

    args = parser.parse_args()
    assert 2011 < args.from_year < 2022, "2011 < args.from_year < 2022"
    assert 2011 < args.to_year < 2022, "2011 < args.to_year < 2022"
    assert args.from_year <= args.to_year, "args.from_year <= args.to_year"
    assert 0 < args.from_month < 13, "0 < args.from_month < 13"
    assert 0 < args.to_month < 13, "0 < args.to_month < 13"
    assert args.from_year < args.to_year or (args.from_year == args.to_year and args.from_month < args.to_month),\
        "args.from_year < args.to_year or (args.from_year == args.to_year and args.from_month < args.to_month)"

    date = datetime(args.from_year, args.from_month, 1)

    while date != datetime(args.to_year + (1 if args.to_month == 12 else 0), 1 if args.to_month == 12 else args.to_month + 1, 1):
        print(f"Fetching zip for date {date.strftime('%Y-%m-%d')}")
        get_zip(date)
        extract_zip(date)
        date = date + timedelta(days=1)


if __name__ == "__main__":
    get_all_zips()

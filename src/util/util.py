import os
import sys

import numpy as np
import requests
from requests import Response


def prep_zeros_if_needed(value: str, number_of_zeros: int):
    for i in range(number_of_zeros - len(value) + 1):
        value = '0' + value
    return value


def declination_of_earth(date):
    day_of_year = date.timetuple().tm_yday
    return 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + day_of_year) / 365.0))


def check_file_status(filepath):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    downloaded = size / (1024 * 1024)
    sys.stdout.write('%.3f MB Downloaded' % downloaded)
    sys.stdout.flush()


def download(output_path: str, url: str):
    print(f"Downloading {url} to path {output_path}")
    response = requests.get(url, stream=True)

    with open(output_path, 'wb') as outfile:
        chunk_size = 1048576
        for chunk in response.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            check_file_status(output_path)

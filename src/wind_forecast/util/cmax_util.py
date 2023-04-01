import math
import os
import sys

from wind_forecast.loaders.CMAXLoader import CMAXLoader

if sys.version_info <= (3, 8, 2):
    import pickle5 as pickle
else:
    import pickle
import re
import numpy as np
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from util.util import prep_zeros_if_needed
from wind_forecast.consts import CMAX_NPY_FILENAME_FORMAT, CMAX_NPY_FILE_REGEX, DATE_KEY_REGEX, CMAX_H5_FILE_REGEX
from wind_forecast.util.logging import log

CMAX_DATASET_DIR = os.environ.get('CMAX_DATASET_DIR')
CMAX_DATASET_DIR = 'data' if CMAX_DATASET_DIR is None else CMAX_DATASET_DIR
CMAX_MAX = 255
CMAX_MIN = 0


def get_available_cmax_h5(from_year: int = 2015, to_year: int = 2022):
    matcher = re.compile(CMAX_H5_FILE_REGEX)
    return [f.name for f in tqdm(os.scandir(CMAX_DATASET_DIR)) if matcher.match(f.name)
            and from_year <= int(matcher.match(f.name).group(1)) < to_year]


def get_available_cmax_hours(from_year: int = 2015, to_year: int = 2022):
    meta_files_matcher = re.compile(r"(\d{4})(\d{2})_meta\.pkl")
    pickle_dir = os.path.join(CMAX_DATASET_DIR, 'pkl')
    log.info(f"Scanning {[pickle_dir]} looking for CMAX meta files.")
    meta_files = [f.name for f in tqdm(os.scandir(pickle_dir)) if meta_files_matcher.match(f.name)
                  and from_year <= int(meta_files_matcher.match(f.name).group(1)) < to_year]
    date_keys = []
    for meta_file in meta_files:
        with open(os.path.join(pickle_dir, meta_file), 'rb') as f:
            date_keys.extend(pickle.load(f))

    date_keys.sort()
    return list(set(date_keys))


def date_from_cmax_npy_file(filename):
    date_matcher = re.match(CMAX_NPY_FILE_REGEX, filename)

    year = int(date_matcher.group(1))
    month = int(date_matcher.group(2))
    day = int(date_matcher.group(3))
    hour = int(date_matcher.group(4))
    minutes = int(date_matcher.group(5))
    date = datetime(year, month, day, hour, minutes)
    return date


def date_from_cmax_date_key(date_key):
    date_matcher = re.match(DATE_KEY_REGEX, date_key)

    year = int(date_matcher.group(1))
    month = int(date_matcher.group(2))
    day = int(date_matcher.group(3))
    hour = int(date_matcher.group(4))
    date = datetime(year, month, day, hour)
    return date


def get_cmax_values_for_sequence(date: pd.Timestamp, cmax_values, sequence_length):
    date = pd.to_datetime(date)
    date_key = CMAXLoader.get_date_key(date)
    values = []
    for frame in range(0, sequence_length):
        values.append(np.array(cmax_values[date_key]))
        date = date + timedelta(hours=1)
        date_key = CMAXLoader.get_date_key(date)

    return values


def get_cmax(id):
    cmax_loader = CMAXLoader()
    values = cmax_loader.get_cmax_image(id)
    return values


# Also return a dictionary of values to have them all read into the runtime
def get_mean_and_std_cmax(cmax_loader: CMAXLoader, dates: [pd.Timestamp], dim: (int, int), sequence_length: int,
                          future_sequence_length: int = 0, prediction_offset: int = 0):
    # Bear in mind that date_keys are indices of FIRST frame in the sequence. Not all frames exist in date_keys because of that fact.
    log.info("Calculating std and mean for the CMAX dataset")
    all_ids = set([item for sublist in [[get_cmax_datekey_from_offset(date, offset) for offset in
                                         range(0, sequence_length + future_sequence_length + prediction_offset)] for
                                        date in dates] for item in sublist])
    mean, sqr_mean = 0, 0
    denom = len(all_ids) * dim[0] * dim[1] / 4
    for id in tqdm(all_ids):
        values = cmax_loader.get_cmax_image(id)
        mean += np.sum(values) / denom
        sqr_mean += np.sum(np.power(values, 2)) / denom

    std = math.sqrt(sqr_mean - pow(mean, 2))

    return mean, std


def get_cmax_datekey_from_offset(date: pd.Timestamp, offset: int) -> str:
    date = pd.to_datetime(date)
    date = date + timedelta(hours=offset)
    return CMAXLoader.get_date_key(date)


def get_min_max_cmax(cmax_loader: CMAXLoader, dates: [pd.Timestamp], sequence_length: int, future_sequence_length: int = 0,
                     prediction_offset: int = 0):
    all_ids = set([item for sublist in [[get_cmax_datekey_from_offset(date, offset) for offset in
                                         range(0, sequence_length + future_sequence_length + prediction_offset)] for
                                        date
                                        in dates] for item in sublist])
    log.info("Loading CMAX files into the runtime.")
    for id in tqdm(all_ids):
        values = cmax_loader.get_cmax_image(id)

    return CMAX_MIN, CMAX_MAX  # We know max and min upfront, let's not waste time :)


def get_cmax_npy_filename(date: datetime):
    return CMAX_NPY_FILENAME_FORMAT.format(date.year, prep_zeros_if_needed(str(date.month), 1),
                                           prep_zeros_if_needed(str(date.day), 1),
                                           prep_zeros_if_needed(str(date.hour), 1),
                                           prep_zeros_if_needed(str(date.minute), 1))


def initialize_synop_dates_for_sequence_with_cmax(cmax_IDs: [str], labels: pd.DataFrame, sequence_length: int,
                                                  future_seq_length: int, prediction_offset: int,
                                                  use_future_cmax: bool = False):
    synop_dates = []
    one_hour = timedelta(hours=1)
    log.info("Preparing sequences of synop and CMAX files.")
    for date in tqdm(labels["date"]):
        cmax_date_key = CMAXLoader.get_date_key(date)
        next_date = date + one_hour
        next_cmax_date_key = CMAXLoader.get_date_key(next_date)
        if len(labels[
                   labels["date"] == next_date]) > 0 and cmax_date_key in cmax_IDs and next_cmax_date_key in cmax_IDs:
            # next frame exists, so the sequence is continued
            synop_dates.append(date)

        elif len(labels[labels["date"] == next_date]) > 0:
            # there is no next frame for CMAX, so the sequence is broken. Remove past frames of sequence_length (and future_length if use_future_cmax)
            for frame in range(0, sequence_length + (
                    0 if not use_future_cmax else prediction_offset + future_seq_length) - (1 if cmax_date_key in cmax_IDs else 0)):
                hours = timedelta(hours=frame)
                date_to_remove = date - hours

                if date_to_remove in synop_dates:
                    synop_dates.remove(date_to_remove)
        else:
            # there is no next frame for synop and/or CMAX , so the sequence is broken. Remove past frames of sequence_length AND future_seq_length
            for frame in range(0, sequence_length + future_seq_length + prediction_offset - (1 if cmax_date_key in cmax_IDs else 0)):
                hours = timedelta(hours=frame)
                date_to_remove = date - hours
                if date_to_remove in synop_dates:
                    synop_dates.remove(date_to_remove)

    return synop_dates

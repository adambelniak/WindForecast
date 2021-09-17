import math
import os
import re
import sys

import h5py
import numpy as np
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm

from wind_forecast.util.common_util import prep_zeros_if_needed
from wind_forecast.consts import CMAX_FILENAME_FORMAT, CMAX_FILE_REGEX
from wind_forecast.util.logging import log


CMAX_DATASET_DIR = os.environ.get('CMAX_DATASET_DIR')
CMAX_DATASET_DIR = 'data' if CMAX_DATASET_DIR is None else CMAX_DATASET_DIR


def get_available_hdf_files_cmax_hours():
    matcher = re.compile(r"\d{10}000000dBZ\.cmax\.h5")
    print(f"Scanning {CMAX_DATASET_DIR} looking for HDF files.")
    return [f.name for f in tqdm(os.scandir(CMAX_DATASET_DIR)) if matcher.match(f.name)]


def date_from_cmax_file(filename):
    date_matcher = re.match(CMAX_FILE_REGEX, filename)

    year = int(date_matcher.group(1))
    month = int(date_matcher.group(2))
    day = int(date_matcher.group(3))
    hour = int(date_matcher.group(4))
    minutes = int(date_matcher.group(5))
    date = datetime(year, month, day, hour, minutes)
    return date


def get_cmax_values_for_sequence(id, sequence_length):
    date = date_from_cmax_file(id)
    values = []
    np_mask_for_cmax = np.load(os.path.join(CMAX_DATASET_DIR, "mask.npy"))
    with h5py.File(os.path.join(CMAX_DATASET_DIR, id), 'r') as hdf:
        data = np.array(hdf.get('dataset1').get('data1').get('data'))
        values.append(data - np_mask_for_cmax)

    for frame in range(1, sequence_length):
        date = date + timedelta(hours=1)
        values.append(get_hdf(id, np_mask_for_cmax))

    return values


def get_hdf(id, mask):
    with h5py.File(os.path.join(CMAX_DATASET_DIR, id), 'r') as hdf:
        data = np.array(hdf.get('dataset1').get('data1').get('data'))
        values = data - mask
    return values


def initialize_mean_and_std_cmax(list_IDs: [str], dim: (int, int), sequence_length: int):
    # Bear in mind that list_IDs are indices of FIRST frame in the sequence. Not all frames exist in list_IDs because of that fact.
    log.info("Calculating std and mean for a CMAX dataset")
    all_ids = set([item for sublist in [[get_cmax_filename_from_offset(id, offset) for offset in range(0, sequence_length)] for id in list_IDs] for item in sublist])
    mean, sqr_mean = 0, 0
    np_mask_for_cmax = np.load(os.path.join(CMAX_DATASET_DIR, "mask.npy"))
    denom = len(all_ids) * dim[0] * dim[1]
    for id in tqdm(all_ids):
        with h5py.File(os.path.join(CMAX_DATASET_DIR, id), 'r') as hdf:
            data = np.array(hdf.get('dataset1').get('data1').get('data'), dtype=float)
            values = data - np_mask_for_cmax
            mean += np.sum(values) / denom
            sqr_mean += np.sum(np.power(values, 2)) / denom

    std = math.sqrt(sqr_mean - pow(mean, 2))

    return mean, std


def get_cmax_filename_from_offset(id: str, offset: int) -> str:
    date = date_from_cmax_file(id)
    date = date + timedelta(hours=offset)
    return get_cmax_filename(date)


def initialize_min_max_cmax(list_IDs: [str], sequence_length: int):
    # Bear in mind that list_IDs are indices of FIRST frame in the sequence. Not all frames exist in list_IDs because of that fact.
    log.info("Calculating min and max for CMAX a dataset")
    all_ids = set([item for sublist in [[get_cmax_filename_from_offset(id, offset) for offset in range(0, sequence_length)] for id in list_IDs] for item in sublist])
    min_val, max_val = sys.float_info.max, sys.float_info.min
    np_mask_for_cmax = np.load(os.path.join(CMAX_DATASET_DIR, "mask.npy"))

    for id in all_ids:
        with h5py.File(os.path.join(CMAX_DATASET_DIR, id), 'r') as hdf:
            data = np.array(hdf.get('dataset1').get('data1').get('data'), dtype=float)
            values = data - np_mask_for_cmax
            min_val = min_val(values, min_val)
            max_val = max_val(values, max_val)

    return min_val, max_val


def get_cmax_filename(date: datetime):
    return CMAX_FILENAME_FORMAT.format(date.year, prep_zeros_if_needed(str(date.month), 1),
                                       prep_zeros_if_needed(str(date.day), 1),
                                       prep_zeros_if_needed(str(date.hour), 1),
                                       prep_zeros_if_needed(str(date.minute), 1))


def initialize_CMAX_list_IDs_and_synop_dates_for_sequence(cmax_IDs: [str], labels: pd.DataFrame, sequence_length: int,
                                                          future_seq_length: int, prediction_offset: int):
    new_list_IDs = []
    synop_dates = []
    one_hour = timedelta(hours=1)
    print("Preparing sequences of synop and HDF files.")
    for date in tqdm(labels["date"]):
        cmax_filename = get_cmax_filename(date)
        if cmax_filename in cmax_IDs:
            next_date = date + one_hour
            next_cmax_filename = get_cmax_filename(next_date)
            if len(labels[labels["date"] == next_date]) > 0 and next_cmax_filename in cmax_IDs and os.path.getsize(os.path.join(CMAX_DATASET_DIR, cmax_filename)) > 0:
                # next frame exists, so the sequence is continued
                synop_dates.append(date)
                new_list_IDs.append(cmax_filename)
            elif len(labels[labels["date"] == next_date]) > 0:
                # there is no next frame for CMAX, so the sequence is broken. Remove past frames of sequence_length
                for frame in range(1, sequence_length):
                    hours = timedelta(hours=frame)
                    date_to_remove = date - hours
                    if date_to_remove in synop_dates:
                        synop_dates.remove(date_to_remove)
                    cmax_filename_to_remove = get_cmax_filename(date_to_remove)
                    if cmax_filename_to_remove in new_list_IDs:
                        new_list_IDs.remove(cmax_filename_to_remove)
            else:
                # there is no next frame for synop and/or CMAX , so the sequence is broken. Remove past frames of sequence_length AND future_seq_length
                for frame in range(1, sequence_length + future_seq_length + prediction_offset):
                    hours = timedelta(hours=frame)
                    date_to_remove = date - hours
                    if date_to_remove in synop_dates:
                        synop_dates.remove(date_to_remove)
                    cmax_filename_to_remove = get_cmax_filename(date_to_remove)
                    if cmax_filename_to_remove in new_list_IDs:
                        new_list_IDs.remove(cmax_filename_to_remove)

    assert len(new_list_IDs) == len(synop_dates)
    return new_list_IDs, synop_dates

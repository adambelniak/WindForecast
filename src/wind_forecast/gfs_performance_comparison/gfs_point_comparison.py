import argparse
import os
import re

from tqdm import tqdm
from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY, NETCDF_FILE_REGEX
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.gfs_util import target_param_to_gfs_name_level, \
    get_available_numpy_files, GFS_DATASET_DIR, date_from_gfs_np_file, get_point_from_GFS_slice_for_coords
import numpy as np


def get_gfs_values_and_targets_for_gfs_ids(available_gfs, labels, target_param, lat, lon):
    targets = []
    gfs_values = []
    coords = Coords(lat, lat, lon, lon)
    param = target_param_to_gfs_name_level(target_param)[0]
    for id in tqdm(available_gfs):
        date = date_from_gfs_np_file(id)
        label = labels[labels["date"] == date]
        if len(label) > 0:
            targets.append(label[target_param].to_numpy())
            gfs_values.append(get_point_from_GFS_slice_for_coords(
                np.load(os.path.join(GFS_DATASET_DIR, param['name'], param['level'], id)), coords))

    return np.array(gfs_values), np.array(targets).squeeze()


def compare_gfs_with_synop(labels, gfs_ids, lat, lon, target_param):
    gfs_data, targets = get_gfs_values_and_targets_for_gfs_ids(gfs_ids, labels, target_param, lat, lon)

    if target_param == 'temperature':
        gfs_data = gfs_data - 273.15  # gfs_data is in Kelvins

    diffs = abs(targets - gfs_data)
    mean_diff = diffs.mean()
    print(mean_diff)
    max_diff = diffs.max()
    print(max_diff)


def main(lat: float, lon: float, target_param: str, synop_file: str, prediction_offset: int, sequence_length: int, from_year: int = 2019):
    synop_data, _, _ = prepare_synop_dataset(synop_file, [target_param], norm=False,
                                             dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=from_year)

    labels = synop_data[['date', target_param]]

    if sequence_length == 1:
        available_gfs = get_available_numpy_files(target_param_to_gfs_name_level(target_param), prediction_offset,
                                                  GFS_DATASET_DIR)
    else:
        available_gfs = []
        for offset in range(prediction_offset, prediction_offset + 3 * sequence_length, 3):
            available_gfs.append(get_available_numpy_files(target_param_to_gfs_name_level(target_param), offset,
                                                           GFS_DATASET_DIR))
        available_gfs = [item for sublist in available_gfs for item in sublist]

    available_gfs = list(filter(lambda item: int(re.match(NETCDF_FILE_REGEX, item).group(1)[:4]) >= from_year, available_gfs))
    compare_gfs_with_synop(labels, available_gfs, lat, lon, target_param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lat", help="Latitude", type=float, default=52.1831174)
    parser.add_argument("--lon", help="Longitude", type=float, default=20.9875259)
    parser.add_argument("--target_param", help="Target parameter", type=str, default="temperature")
    parser.add_argument("--synop_file", help="File with synop observations", type=str,
                        default="WARSZAWA-OKECIE_375_data.csv")
    parser.add_argument("--prediction_offset", help="Prediction offset", type=int, default=3)
    parser.add_argument("--sequence_length", help="Length of predicted sequence", type=int, default=8)
    parser.add_argument("--from_year", help="Include dates from this year forward", type=int, default=2018)

    args = parser.parse_args()

    main(**vars(args))

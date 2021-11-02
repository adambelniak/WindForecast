import argparse

import numpy as np
from tqdm import tqdm

from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY, STATION_META
from wind_forecast.loaders.GFSLoader import GFSLoader
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.gfs_util import target_param_to_gfs_name_level, \
    get_available_gfs_date_keys, get_point_from_GFS_slice_for_coords, date_from_gfs_date_key


def get_gfs_values_and_targets_for_gfs_ids(gfs_date_keys, labels, target_param, lat: float, lon: float, offset: int):
    targets = []
    gfs_values = []
    coords = Coords(lat, lat, lon, lon)
    gfs_loader = GFSLoader()
    param = target_param_to_gfs_name_level(target_param)[0]
    for date_key in tqdm(gfs_date_keys):
        date = date_from_gfs_date_key(date_key)
        label = labels[labels["date"] == date]
        if len(label) > 0:
            targets.append(label[target_param].to_numpy())
            gfs_values.append(
                get_point_from_GFS_slice_for_coords(gfs_loader.get_gfs_image(date_key, param, offset), coords))

    return np.array(gfs_values), np.array(targets).squeeze()


def compare_gfs_with_synop(labels, gfs_date_keys, target_param, lat: float, lon: float, prediction_offset: int,
                           sequence_length: int):
    gfs_data, targets = np.array([]), np.array([])
    for offset in range(prediction_offset, prediction_offset + 3 * sequence_length, 3):
        next_gfs_data, next_targets = get_gfs_values_and_targets_for_gfs_ids(gfs_date_keys[str(offset)], labels,
                                                                             target_param, lat, lon, offset)
        if target_param == 'temperature':
            next_gfs_data = next_gfs_data - 273.15  # gfs_data is in Kelvins

        gfs_data = np.append(gfs_data, next_gfs_data)
        targets = np.append(targets, next_targets)

    diffs = abs(targets - gfs_data)
    mean_diff = diffs.mean()
    print(mean_diff)
    max_diff = diffs.max()
    print(max_diff)


def main(station: str, target_param: str, prediction_offset: int, sequence_length: int, from_year: int):
    synop_file = STATION_META[station]['synop_file']
    synop_data = prepare_synop_dataset(synop_file, [target_param], norm=False,
                                             dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=from_year)

    labels = synop_data[['date', target_param]]

    available_gfs_date_keys = get_available_gfs_date_keys(target_param_to_gfs_name_level(target_param),
                                                          prediction_offset, sequence_length, from_year)

    compare_gfs_with_synop(labels, available_gfs_date_keys, target_param, STATION_META[station]['lat'],
                           STATION_META[station]['lon'], prediction_offset, sequence_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--station", help="Name of synoptic station", type=str, default='Warsaw')
    parser.add_argument("--target_param", help="Target parameter", type=str, default="temperature")
    parser.add_argument("--prediction_offset", help="Prediction offset", type=int, default=0)
    parser.add_argument("--sequence_length", help="Length of predicted sequence", type=int, default=8)
    parser.add_argument("--from_year", help="Include dates from this year forward", type=int, default=2017)

    args = parser.parse_args()

    main(**vars(args))

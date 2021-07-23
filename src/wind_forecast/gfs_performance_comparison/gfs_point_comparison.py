import argparse
import os

from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset
from wind_forecast.util.utils import match_gfs_with_synop_sequence


def main(lat: float, lon: float, target_param: str, synop_file: str, prediction_offset: int):
    synop_data, _, _ = prepare_synop_dataset(synop_file, [target_param], norm=False,
                                           dataset_dir=SYNOP_DATASETS_DIRECTORY, from_year=2015)

    labels = synop_data[['date', target_param]].to_numpy()

    targets = [labels[i + prediction_offset - 1] for i in
               range(labels.shape[0] - prediction_offset + 1)]

    _, gfs_data, targets = match_gfs_with_synop_sequence(targets, targets, lat, lon, prediction_offset, target_param, exact_date_match=True)
    if target_param == 'temperature':
        gfs_data = gfs_data - 273.15  # gfs_data is in Kelvins
    mean_diff = abs(targets - gfs_data).mean()
    print(mean_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lat", help="Latitude", type=float, default=52.1831174)
    parser.add_argument("--lon", help="Longitude", type=float, default=20.9875259)
    parser.add_argument("--target_param", help="Target parameter", type=str, default="wind_velocity")
    parser.add_argument("--synop_file", help="File with synop observations", type=str,
                        default="KOZIENICE_488_data.csv")
    parser.add_argument("--prediction_offset", help="Prediction offset", type=int, default=3)

    args = parser.parse_args()

    main(**vars(args))

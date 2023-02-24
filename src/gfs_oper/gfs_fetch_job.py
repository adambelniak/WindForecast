import argparse

from gfs_oper.common import Config
from gfs_oper.fetch_oper import fetch_recent_gfs
from gfs_oper.process_oper import process_recent_gfs
from util.coords import Coords

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lat', help='Latitude of target station.', default=None, type=float)
    parser.add_argument('--lon', help='Longitude of target station.', default=None, type=float)
    parser.add_argument('--past_sequence_length', help='Number of hours for past forecasts to fetch',
                        default=24, type=int)
    parser.add_argument('--future_sequence_length', help='Number of hours for future forecasts to fetch',
                        default=24, type=int)

    args = parser.parse_args()
    config = Config(args.past_sequence_length, args.future_sequence_length,
                            "download", "processed", Coords(args.lat, args.lat, args.lon, args.lon))
    fetch_recent_gfs(config)
    process_recent_gfs(config)

import argparse
import os
import time
from typing import Union

import schedule

from gfs_oper.common import Config
from gfs_oper.fetch_oper import fetch_recent_gfs, get_init_meta_from_init_string, GribResource
from gfs_oper.process_oper import process_recent_gfs
from util.coords import Coords
import pandas as pd


def fetch_oper_gfs(config: Config) -> Union[pd.DataFrame, None]:
    fetch_recent_gfs(config)
    latest_ready_forecast = process_recent_gfs(config)
    if latest_ready_forecast is None:
        return None
    init_meta = get_init_meta_from_init_string(latest_ready_forecast)

    data = None
    for offset in range(0, config.future_sequence_length):
        grib = GribResource(init_meta, offset)
        csv_path = os.path.join(os.path.dirname(grib.get_output_location(config.processing_output_path)),
                               f"{str(config.target_coords.nlat)}-{str(config.target_coords.wlon)}-{str(grib.offset)}.csv")
        df = pd.read_csv(csv_path)
        if data is None:
            data = pd.DataFrame(columns=df.columns)
        data = pd.concat([data, df])

    for init in range(0, (config.past_sequence_length // 6) + 1):
        init_meta = init_meta.get_previous()

        for offset in range(0 if init < (config.past_sequence_length // 6) else (6 - config.past_sequence_length % 6), 6):
            grib = GribResource(init_meta, offset)
            csv_path = os.path.join(os.path.dirname(grib.get_output_location(config.processing_output_path)),
                                    f"{str(config.target_coords.nlat)}-{str(config.target_coords.wlon)}-{str(grib.offset)}.csv")
            df = pd.read_csv(csv_path)
            data = pd.concat([data, df])

    data = data.sort_values(by='date').reset_index().drop(['index'], axis=1)
    return data


def main():
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
    try:
        job = schedule.every(1).hours.do(lambda: fetch_oper_gfs(config))
        job.run()
        while True:
            schedule.run_pending()
            time.sleep(60)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
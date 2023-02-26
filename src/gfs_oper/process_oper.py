import os
from pathlib import Path
from typing import Union

import pygrib
from datetime import datetime, timedelta

from scipy.interpolate import interpolate

from gfs_archive_0_25.utils import get_nearest_coords
from gfs_common.common import get_param_key
from gfs_oper.common import Config, InitMeta
from gfs_oper.fetch_oper import get_init_meta, GribResource
from util.coords import Coords
import pandas as pd

from wind_forecast.util.config import process_config


def process_grib(grib: GribResource, grib_dir: str, target_coords: Coords, output_path: str):
    output_file = os.path.join(os.path.dirname(grib.get_output_location(output_path)),
                               f"{str(target_coords.nlat)}-{str(target_coords.wlon)}-{str(grib.offset)}.csv")
    if os.path.exists(output_file):
        print(f"Skipping grib processing, because file at {output_file} already exists")
        return
    print(f"Processing grib {grib.get_output_location(grib_dir)}...")

    param_config = process_config(os.path.join(Path(__file__).parent, "param_config.json")).params
    data = {}

    gr = pygrib.open(grib.get_output_location(grib_dir))

    for parameter in param_config:
        try:
            if 'type_of_level' in parameter:
                message = gr.select(shortName=parameter['grib_name'], typeOfLevel=parameter['type_of_level'],
                                    level=parameter['grib_level'])[0]
            else:
                message = gr.select(shortName=parameter['grib_name'])[0]
            coords = get_nearest_coords(target_coords)
            values = message.data(lat1=coords.slat,
                                  lat2=coords.nlat,
                                  lon1=coords.wlon,
                                  lon2=coords.elon)
            interpolated_function_for_values = interpolate.interp2d([coords.slat, coords.nlat], [coords.wlon, coords.elon], values[0][::-1])
            data[get_param_key(parameter['name'], parameter['level'])] = interpolated_function_for_values(target_coords.nlat, target_coords.wlon).item()
        except:
            print(f"Data not found in grib file for parameter {parameter['name']}. Setting it to 0.0.")
            data[get_param_key(parameter['name'], parameter['level'])] = 0.0
    gr.close()
    data['date'] = (grib.init_meta.get_date_string_for_offset(grib.offset))
    df = pd.DataFrame([data])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)


def process_future_gribs(init_meta: InitMeta, config: Config):
    for offset in range(0, config.future_sequence_length):
        process_grib(GribResource(init_meta, offset), config.download_path, config.target_coords, config.processing_output_path)


def process_past_gribs(init_meta, config: Config):
    for init in range(0, (config.past_sequence_length // 6) + 1):
        init_meta = init_meta.get_previous()
        for offset in range(0, 6):
            process_grib(GribResource(init_meta, offset), config.download_path, config.target_coords,
                         config.processing_output_path)


def process_all_needed_gribs(init_meta: InitMeta, config: Config) -> None:
    process_future_gribs(init_meta, config)
    process_past_gribs(init_meta, config)


def save_info(init_meta, processing_output_path):
    os.makedirs(os.path.dirname(available_starting_points_path(processing_output_path)), exist_ok=True)
    with open(available_starting_points_path(processing_output_path), 'a') as f:
        f.write(init_meta.get_date_string() + "\n")


def available_starting_points_path(processing_output_path: str) -> str:
    return os.path.join(Path(__file__).parent, processing_output_path, "available_starting_points.txt")


def is_forecast_ready(init_meta: InitMeta, processing_output_path: str) -> Union[str, None]:
    records_file = available_starting_points_path(processing_output_path)
    if os.path.exists(records_file):
        with open(records_file, 'r') as f:
            for line in f:
                if init_meta.get_date_string() + "\n" == line:
                    return init_meta.get_date_string()

    return None


"""
Processes the most recent available forecast.
Returns the string indicating the run of the forecast or None if none available for 5 recent runs
"""
def process_recent_gfs(config: Config) -> Union[str, None]:
    current_date = datetime.now()
    init_meta = get_init_meta(current_date)
    past_init_meta = get_init_meta(current_date)

    latest_ready_forecast = is_forecast_ready(init_meta, config.processing_output_path)
    if latest_ready_forecast is not None:
        return latest_ready_forecast

    forecast_available = False
    forecast_ready = False
    tries = 0

    while not forecast_available and tries < 5:
        forecast_available = True
        for offset in range(0, config.future_sequence_length):
            if not os.path.exists(GribResource(init_meta, offset).get_output_location(config.download_path)):
                latest_ready_forecast = is_forecast_ready(init_meta, config.processing_output_path)
                if latest_ready_forecast is not None:
                    forecast_ready = True
                    break
                forecast_available = False
                tries += 1
                # get previous GFS run
                current_date = current_date - timedelta(hours=6)
                init_meta = init_meta.get_previous()
                past_init_meta = init_meta
                break
        if forecast_ready:
            break
        if not forecast_available:
            continue

        for init in range(0, (config.past_sequence_length // 6) + 1):
            past_init_meta = past_init_meta.get_previous()

            for offset in range(0, 6):
                if not os.path.exists(GribResource(past_init_meta, offset).get_output_location(config.download_path)):
                    forecast_available = False
                    tries += 1
                    # get previous GFS run
                    current_date = current_date - timedelta(hours=6)
                    init_meta = init_meta.get_previous()
                    past_init_meta = init_meta
                    break
            if not forecast_available:
                break

    if forecast_ready:
        return latest_ready_forecast

    if forecast_available:
        process_all_needed_gribs(init_meta, config)
        save_info(init_meta, config.processing_output_path)
        return init_meta.get_date_string()

    return None


if __name__ == "__main__":
    process_recent_gfs(Config(24, 1, "download", "processed", Coords(52.1831174, 52.1831174, 20.9875259, 20.9875259)))
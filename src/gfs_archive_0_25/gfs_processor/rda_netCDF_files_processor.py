import argparse
import glob
import time
from os.path import isfile, join

import netCDF4 as nc
import re
from datetime import datetime, timedelta
import pandas as pd
import schedule
import tqdm
import numpy as np

from gfs_common.common import GFS_PARAMETERS
from gfs_archive_0_25.gfs_processor.Coords import Coords
from wind_forecast.preprocess.gfs.gfs_preprocess_netCDF import get_values_as_numpy_arr_from_file
from gfs_archive_0_25.gfs_processor.own_logger import logger
from gfs_archive_0_25.utils import prep_zeros_if_needed
from gfs_archive_0_25.gfs_processor.consts import *

OFFSET = 3
POLAND_NLAT = 56
POLAND_SLAT = 48
POLAND_WLON = 13
POLAND_ELON = 26

class ProcessingException(Exception):
    pass


def get_forecast_df_for_date_and_run(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=[0])
    else:
        return pd.DataFrame(columns=["date"])


def get_date_run_offset_from_netCDF_file_name(file):
    date_matcher = re.match(RAW_NETCDF_FILENAME_REGEX, file)
    if date_matcher is None:
        raise ProcessingException(f"Filename {file} does not match raw netCDF file regex: {RAW_NETCDF_FILENAME_REGEX}")

    date_from_filename = date_matcher.group(2)
    year = date_from_filename[:4]
    month = date_from_filename[4:6]
    day = date_from_filename[6:8]
    return datetime(int(year), int(month), int(day)), date_from_filename[8:10], date_matcher.group(3)


def get_value_from_netCDF(dataset, lat_index, lon_index):
    return next(iter(dataset.variables.values()))[0][lat_index][lon_index]


def process_netCDF_file_to_csv(file, param, level):
    ds = nc.Dataset(os.path.join("download", "netCDF", param, level, file))
    lats, lons = ds.variables['lat_0'][:], ds.variables['lon_0'][:]
    init_date, run, offset = get_date_run_offset_from_netCDF_file_name(file)
    date_str = (init_date + timedelta(hours=int(offset))).isoformat()

    for lat_index, lat in enumerate(lats):
        for lon_index, lon in list(enumerate(lons)):
            final_csv_dir = os.path.join("output_csvs", str(lat).replace('.', '_') + '-' + str(lon).replace('.', '_'))
            if not os.path.exists(final_csv_dir):
                os.makedirs(final_csv_dir)

            value = get_value_from_netCDF(ds, lat_index, lon_index)
            param_name = param + '_' + level.replace("_", ":")

            final_csv_name = FINAL_CSV_FILENAME_FORMAT.format(init_date.year,
                                                              prep_zeros_if_needed(str(init_date.month), 1),
                                                              prep_zeros_if_needed(str(init_date.day), 1),
                                                              run)
            final_csv_path = os.path.join(final_csv_dir, final_csv_name)
            forecast_df = get_forecast_df_for_date_and_run(final_csv_path)

            if param_name not in forecast_df:
                forecast_df[param_name] = 0

            rows = forecast_df[forecast_df['date'] == date_str]
            if len(rows) == 0:
                forecast_df.loc[len(forecast_df.index), :] = ''
                forecast_df.loc[len(forecast_df.index) - 1, ['date', param_name]] = [date_str,
                                                                                     value]
                forecast_df.sort_values('date')
            else:
                forecast_df.loc[forecast_df['date'] == date_str, param_name] = value

            forecast_df.to_csv(final_csv_path)
    # #os.remove(csv_file_path)


def process_netCDF_files_to_csv():
    logger.info("Processing netCDF files...")
    netCDF_dir = "download/netCDF"
    for root, param_dirs, filenames in os.walk(netCDF_dir):
        for param_dir in param_dirs:
            logger.info(f"Processing {param_dir} parameter...")
            for root, level_dirs, filenames in os.walk(os.path.join(netCDF_dir, param_dir)):
                for level_dir in level_dirs:
                    logger.info(f"Processing {level_dir} level...")
                    files_in_directory = [f for f in os.listdir(os.path.join(netCDF_dir, param_dir, level_dir)) if
                                          isfile(join(netCDF_dir, param_dir, level_dir, f))]
                    for file in tqdm.tqdm(files_in_directory):
                        file_name_match = re.match(RAW_NETCDF_FILENAME_REGEX, file)
                        if file_name_match is not None:
                            process_netCDF_file_to_csv(file, param_dir, level_dir)
                # do not take subdirs
                break
        # do not take subdirs
        break

    logger.info("Processing done.")


def check_if_any_file_for_year_exists(year, parameter_level_tuple):
    netCDF_file_glob = RDA_NETCDF_FILENAME_FORMAT.format(str(year),
                                                         '*',
                                                         '',
                                                         '',
                                                         '00' + str(OFFSET),
                                                         '*')
    netCDF_path_glob = os.path.join(NETCDF_DOWNLOAD_PATH, parameter_level_tuple[0],
                                    parameter_level_tuple['level'].replace(":", "_").replace(",", "-"), netCDF_file_glob)
    found_files = glob.glob(netCDF_path_glob)
    if len(found_files) > 0:
        return True
    return False


def process_to_numpy_array(parameter_level_tuple, coords: Coords, output_dir: str):
    download_dir = os.path.join(NETCDF_DOWNLOAD_PATH, parameter_level_tuple['name'], parameter_level_tuple['level'])

    for root, dirs, filenames in os.walk(download_dir):
        if len(filenames) > 0:
            output_dir = os.path.join(output_dir, f"\\{parameter_level_tuple['name']}\\{parameter_level_tuple['level']}")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for file in tqdm.tqdm(filenames):
                try:
                    date, run, offset = get_date_run_offset_from_netCDF_file_name(file)

                    output_path = os.path.join(output_dir, FINAL_NUMPY_FILENAME_FORMAT.format(
                        str(date.year),
                        prep_zeros_if_needed(str(date.month), 1),
                        prep_zeros_if_needed(str(date.day), 1),
                        run,
                        prep_zeros_if_needed(str(offset), 2)))

                    # consider as done if output file already exists
                    if not os.path.exists(output_path):
                        forecast = get_values_as_numpy_arr_from_file(os.path.join(download_dir, file), coords)
                        if forecast is not None:
                            np.save(output_path, forecast)
                except ProcessingException:
                    pass
        break


def process_netCDF_files_to_npy(output_dir: str):
    for param in GFS_PARAMETERS:
        logger.info(f"Converting parameter {param['name']} {param['level']}")
        process_to_numpy_array(param, Coords(56, 48, 13, 26), output_dir)


def schedule_processing():
    try:
        job = schedule.every(15).minutes.do(lambda: process_netCDF_files_to_csv())
        job.run()
        while True:
            schedule.run_pending()
            time.sleep(60)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='CSV or NPY - output format to which save the data. ', default="NPY")
    parser.add_argument('--output_dir', help='Output directory for the result files.', default="D:\\WindForecast\\output_np2")
    args = parser.parse_args()

    if args.mode == "CSV":
        schedule_processing()
    elif args.mode == "NPY":
        process_netCDF_files_to_npy(args.output_dir)
    else:
        raise Exception("Unexpected mode: " + args.mode + ". Please specify 'CSV' or 'NPY'")

import argparse
import glob

import netCDF4 as nc
import re
from datetime import datetime, timedelta
import pandas as pd
import tqdm
import numpy as np

from gfs_common.common import GFS_PARAMETERS, GFS_SPACE
from gfs_archive_0_25.gfs_processor.Coords import Coords
from gfs_archive_0_25.gfs_processor.own_logger import logger
from gfs_archive_0_25.utils import prep_zeros_if_needed
from gfs_archive_0_25.gfs_processor.consts import *

OFFSET = 3


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
        logger.warn(f"Filename {file} does not match raw netCDF file regex: {RAW_NETCDF_FILENAME_REGEX}")
        raise ProcessingException(f"Filename {file} does not match raw netCDF file regex: {RAW_NETCDF_FILENAME_REGEX}")

    date_from_filename = date_matcher.group(2)
    year = date_from_filename[:4]
    month = date_from_filename[4:6]
    day = date_from_filename[6:8]
    return datetime(int(year), int(month), int(day)), date_from_filename[8:10], date_matcher.group(3)


def get_values_as_numpy_arr_from_file(netCDF_filepath, coords: Coords):
    ds = nc.Dataset(netCDF_filepath)
    lats, lons = ds.variables[[key for key in ds.variables.keys() if key.startswith('lat')][0]][:],\
                 ds.variables[[key for key in ds.variables.keys() if key.startswith('lon')][0]][:]
    if coords.nlat not in lats or coords.slat not in lats or coords.wlon not in lons or coords.elon not in lons:
        return None
    else:
        lats = lats.data.tolist()
        lons = lons.data.tolist()
        nlat_index, slat_index, wlon_index, elon_index = lats.index(coords.nlat), lats.index(coords.slat), \
                                                         lons.index(coords.wlon), lons.index(coords.elon)

        return ds.variables[[key for key in ds.variables.keys() if key[0].isupper()][0]][nlat_index:slat_index + 1, wlon_index:elon_index + 1].data


def get_value_from_netCDF(dataset, lat_index, lon_index):
    return next(iter(dataset.variables.values()))[0][lat_index][lon_index]


def process_netCDF_file_to_csv(file, parameter_level_tuple, output_dir):
    name, level = parameter_level_tuple
    ds = nc.Dataset(file)
    lats, lons = ds.variables['lat_0'][:], ds.variables['lon_0'][:]
    init_date, run, offset = get_date_run_offset_from_netCDF_file_name(file)
    date_str = (init_date + timedelta(hours=int(offset))).isoformat()

    for lat_index, lat in enumerate(lats):
        for lon_index, lon in list(enumerate(lons)):
            final_csv_dir = os.path.join(output_dir, str(lat).replace('.', '_') + '-' + str(lon).replace('.', '_'))
            if not os.path.exists(final_csv_dir):
                os.makedirs(final_csv_dir)

            value = get_value_from_netCDF(ds, lat_index, lon_index)
            param_name = name + '_' + level.replace("_", ":")

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


def process_to_numpy_array(parameter_level_tuple, input_dir: str, output_dir: str):
    for root, dirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            output_dir = os.path.join(output_dir, parameter_level_tuple['name'], parameter_level_tuple['level'])

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
                        forecast = get_values_as_numpy_arr_from_file(os.path.join(input_dir, file), GFS_SPACE)
                        if forecast is not None:
                            np.save(output_path, forecast)
                except ProcessingException:
                    pass
        break


def process_to_csv(parameter_level_tuple, input_dir: str, output_dir: str):
    for root, level_dirs, filenames in os.walk(input_dir):
        if len(filenames) > 0:
            output_dir = os.path.join(output_dir, parameter_level_tuple['name'], parameter_level_tuple['level'])

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for file in tqdm.tqdm(filenames):
                file_name_match = re.match(RAW_NETCDF_FILENAME_REGEX, file)
                if file_name_match is not None:
                    process_netCDF_file_to_csv(os.path.join(input_dir, file), parameter_level_tuple, output_dir)


def process_netCDF_files_to_csv(output_dir: str):
    for param in GFS_PARAMETERS:
        logger.info(f"Converting parameter {param['name']} {param['level']}")
        process_to_csv(param, GFS_SPACE, output_dir)


def process_netCDF_files(mode: str, output_dir: str):
    for param in GFS_PARAMETERS:
        name, level = param
        logger.info(f"Converting parameter {name} {level}")
        if name == 'T CDC':
            param_name_dir = 'L CDC' if level == 'LCY_0' else 'M CDC' if level == 'MCY_0' else 'H CDC'
        else:
            param_name_dir = name
        input_dir = os.path.join(NETCDF_DOWNLOAD_PATH, param_name_dir, level)

        if mode == "NPY":
            process_to_numpy_array(param, input_dir, output_dir)
        else:
            process_to_csv(param, input_dir, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', help='CSV or NPY - output format to which save the data. ', default="NPY", choices=["CSV", "NPY"])
    parser.add_argument('--output_dir', help='Output directory for the result files.', default="D:\\WindForecast\\gfs")
    args = parser.parse_args()

    process_netCDF_files(args.mode, args.output_dir)

from datetime import datetime, timedelta
import glob
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
from tqdm import tqdm

from gfs_archive_0_25.gfs_processor.consts import RDA_NETCDF_FILENAME_FORMAT, FINAL_NUMPY_FILENAME_FORMAT
from gfs_archive_0_25.utils import prep_zeros_if_needed

NETCDF_DIR = os.path.join("D:\\", "WindForecast", "download", "netCDF")


def get_netCDF_file(init_date, run, offset, param, level):
    netCDF_file_glob = RDA_NETCDF_FILENAME_FORMAT.format(str(init_date.year),
                                                         prep_zeros_if_needed(str(init_date.month), 1),
                                                         prep_zeros_if_needed(str(init_date.day), 1),
                                                         str(run),
                                                         prep_zeros_if_needed(str(offset), 2),
                                                         "*")
    netCDF_path_glob = os.path.join(NETCDF_DIR, param, level.replace(":", "_").replace(",", "-"), netCDF_file_glob)
    found_files = glob.glob(netCDF_path_glob)
    if len(found_files) > 0:
        return found_files
    return None


def get_values_as_numpy_arr_from_file(netCDF_filepath, nlat, slat, wlon, elon):
    ds = nc.Dataset(netCDF_filepath)
    lats, lons = ds.variables['lat_0'][:], ds.variables['lon_0'][:]
    if nlat not in lats or slat not in lats or wlon not in lons or elon not in lons:
        return None
    else:
        lats = lats.data.tolist()
        lons = lons.data.tolist()
        nlat_index, slat_index, wlon_index, elon_index = lats.index(nlat), lats.index(slat), lons.index(wlon), lons.index(elon)

        return next(iter(ds.variables.values()))[0, nlat_index:slat_index + 1, wlon_index:elon_index + 1].data


def create_single_slice_for_param_and_region(init_date, run, offset, param, level, nlat, slat, wlon, elon):
    netCDF_filepaths = get_netCDF_file(init_date, run, offset, param, level)
    if netCDF_filepaths is None:
        return np.zeros(((nlat - slat) * 4 + 1, (elon - wlon) * 4 + 1))

    values = None
    for file in netCDF_filepaths:
        values = get_values_as_numpy_arr_from_file(file, nlat, slat, wlon, elon)
        if values is not None:
            break

    if values is None:
        raise Exception(f"Specified region is not fully covered by any of netCDF files found.")
    return values


def get_forecast_for_date_and_params(init_date, run, offset, param_level_tuples: [(str, str)], nlat, slat, wlon, elon):
    result = []
    for (param, level) in param_level_tuples:
        result.append(create_single_slice_for_param_and_region(init_date, run, offset, param, level, nlat, slat, wlon, elon))

    return np.array(result)


def get_forecasts_for_date_offsets_and_params(init_date, run, init_offset, end_offset, param_level_tuples: [(str, str)], nlat, slat, wlon, elon):
    if init_offset == end_offset:
        return get_forecast_for_date_and_params(init_date, run, init_offset, param_level_tuples, nlat, slat, wlon, elon)

    if init_offset % 3 != 0:
        init_offset = init_offset + (3 - init_offset % 3)
    if end_offset % 3 != 0:
        end_offset = end_offset + (3 - end_offset % 3)

    result = []
    for offset in range(init_offset, end_offset, 3):
        result.append(get_forecast_for_date_and_params(init_date, run, offset, param_level_tuples, nlat, slat, wlon, elon))

    return np.array(result)


def get_forecasts_for_date_all_runs_specified_offsets_and_params(init_date, init_offset, end_offset, param_level_tuples: [(str, str)], nlat, slat, wlon, elon):
    result = []
    for run in ['00', '06', '12', '18']:
        result.append(get_forecasts_for_date_offsets_and_params(init_date, run, init_offset, end_offset, param_level_tuples, nlat, slat, wlon, elon))

    return np.array(result)


def get_forecasts_for_date_range_all_runs_specified_offsets_and_params(init_date, end_date, init_offset, end_offset, param_level_tuples: [(str, str)], nlat, slat, wlon, elon):
    result = []
    date = init_date
    delta = end_date - init_date
    for i in tqdm(range(delta.days)):
        result.append(get_forecasts_for_date_all_runs_specified_offsets_and_params(date, init_offset, end_offset, param_level_tuples, nlat, slat, wlon,
                                                                                   elon))
        date = date + timedelta(days=1)

    return np.array(result)


def get_forecasts_for_year_offset_param_from_npy_file(year, param_level_tuple, offset, dataset_dir):
    if year == 2015:
        init_date = datetime(2015, 1, 15)
    else:
        init_date = datetime(year, 1, 1)
    end_date = datetime(init_date.year + 1, 1, 1)

    netCDF_filename = FINAL_NUMPY_FILENAME_FORMAT.format(init_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                                                         prep_zeros_if_needed(str(offset), 2))

    netCDF_path = os.path.join(dataset_dir, param_level_tuple[0], param_level_tuple[1], netCDF_filename)

    with open(netCDF_path, 'rb') as f:
        return np.load(f)



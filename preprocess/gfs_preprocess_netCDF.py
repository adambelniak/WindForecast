import glob
import os
import netCDF4 as nc
import numpy as np

from gfs_archive_0_25.gfs_processor.consts import RDA_NETCDF_FILENAME_FORMAT


def get_netCDF_file(init_date, run, offset, param, level):
    netCDF_file_glob = RDA_NETCDF_FILENAME_FORMAT.format(str(init_date.year),
                                                         str(init_date.month),
                                                         str(init_date.day),
                                                         str(run),
                                                         offset,
                                                         "*")
    netCDF_path_glob = os.path.join("..", "gfs_archive_0_25", "gfs_processor", "download", "netCDF", param,
                                    level.replace(":", "_").replace(",", "-"), netCDF_file_glob)
    found_files = glob.glob(netCDF_path_glob)
    if found_files is not None:
        return found_files
    return None


def get_values_as_numpy_arr_from_file(netCDF_filepath, nlat, slat, wlon, elon):
    ds = nc.Dataset(netCDF_filepath)
    lats, lons = ds.variables['lat_0'][:], ds.variables['lon_0'][:]
    if nlat not in lats or slat not in lats or wlon not in lons or elon not in lons:
        return None
    else:
        nlat_index, slat_index, wlon_index, elon_index = lats.index(nlat), lats.index(slat), lons.index(wlon), lons.index(elon)
        return np.array(next(iter(ds.variables.values()))[0, slat_index:nlat_index, wlon_index, elon_index])


def create_single_slice_for_param_and_region(init_date, run, offset, param, level, nlat, slat, wlon, elon):
    netCDF_filepaths = get_netCDF_file(init_date, run, offset, param, level)
    if netCDF_filepaths is None:
        raise Exception(f"Could not find netCDF file for init date {init_date.isoformat()}, run {run}, offset {offset},"
                        f" param {param}, level {level}.")

    values = None
    for file in netCDF_filepaths:
        if values is not None:
            values = get_values_as_numpy_arr_from_file(file, nlat, slat, wlon, elon)
            break

    if values is None:
        raise Exception(f"Specified region is not fully covered by any of netCDF files found.")
    return values


def get_forecast_for_date_and_params(init_date, run, offset, param_level_tuples, nlat, slat, wlon, elon):
    result = []
    for tuple in param_level_tuples:
        result.append(create_single_slice_for_param_and_region(init_date, run, offset, tuple[0], tuple[1], nlat, slat, wlon, elon))

    return np.array(result)


def get_forecasts_for_date_offsets_and_params(init_date, run, init, end, param_level_tuples, nlat, slat, wlon, elon):
    if init % 3 != 0:
        init = init + (3 - init % 3)
    if end % 3 != 0:
        end = end - (end % 3)

    result = []
    for offset in range(init, end, 3):
        result.append(get_forecast_for_date_and_params(init_date, run, offset, param_level_tuples, nlat, slat, wlon, elon))

    return np.array(result)


def get_forecasts_for_date_all_runs_offsets_and_params(init_date, init, end, param_level_tuples, nlat, slat, wlon, elon):
    result = []
    for run in ['00', '06', '12', '18']:
        result.append(get_forecasts_for_date_offsets_and_params(init_date, run, init, end, param_level_tuples, nlat, slat, wlon, elon))

    return np.array(result)


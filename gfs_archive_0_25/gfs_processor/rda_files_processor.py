from os.path import isfile, join
import netCDF4 as nc
import os
import re
import datetime
import sys
import pandas as pd

sys.path.insert(1, '../..')

from gfs_archive_0_25.utils import prep_zeros_if_needed
from gfs_archive_0_25.gfs_processor.consts import *


def get_value_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, error_bad_lines=False, warn_bad_lines=False,
                     names=['#ParameterName', 'Longitude', 'Latitude', 'ValidDate', 'ValidTime', 'LevelValue',
                            'ParameterValue'])
    value = df.iloc[1]['ParameterValue']
    if value == 'lev':  # in some csvs first row is different
        value = df.iloc[2]['ParameterValue']
    return value


def prepare_final_csvs_from_csvs(dir_with_csvs, latitude: str, longitude: str):
    final_date = datetime.datetime.now()
    latlon_dir = os.path.join("download/csv", dir_with_csvs)
    final_csv_dir = os.path.join("output_csvs", latitude.replace('.', '_') + '-' + longitude.replace('.', '_'))
    if not os.path.exists(final_csv_dir):
        os.makedirs(final_csv_dir)

    init_date = datetime.datetime(2015, 1, 15)
    while init_date < final_date:
        for run in ['00', '06', '12', '18']:
            # final csv name for this forecast
            final_csv_name = FINAL_CSV_FILENAME_FORMAT.format(init_date.year,
                                                              prep_zeros_if_needed(str(init_date.month), 1),
                                                              prep_zeros_if_needed(str(init_date.day), 1),
                                                              run)
            final_csv_path = os.path.join(final_csv_dir, final_csv_name)
            if os.path.exists(final_csv_path):
                forecast_df = pd.read_csv(final_csv_path, index_col=[0])
            else:
                forecast_df = pd.DataFrame(columns=["date"])
            offset = 3
            while offset < 168:
                # fetch values from all params and levels
                for root, param_dirs, filenames in os.walk(latlon_dir):
                    for param_dir in param_dirs:
                        for root, level_dirs, filenames in os.walk(os.path.join(latlon_dir, param_dir)):
                            for level_dir in level_dirs:
                                level_dir_path = os.path.join(latlon_dir, param_dir, level_dir)
                                csv_file_name = RDA_CSV_FILENAME_FORMAT.format(init_date.year,
                                                                               prep_zeros_if_needed(str(init_date.month), 1),
                                                                               prep_zeros_if_needed(str(init_date.day), 1),
                                                                               run,
                                                                               prep_zeros_if_needed(str(offset), 2),
                                                                               'belniak')
                                csv_file_path = os.path.join(level_dir_path, csv_file_name)

                                if os.path.exists(csv_file_path):
                                    value = get_value_from_csv(csv_file_path)
                                    param_name = param_dir + '_' + level_dir
                                    date = (init_date + datetime.timedelta(hours=offset)).isoformat()

                                    if param_name not in forecast_df:
                                        forecast_df[param_name] = 0

                                    rows = forecast_df[forecast_df['date'] == date]
                                    if len(rows) == 0:
                                        forecast_df.loc[len(forecast_df.index), :] = ''
                                        forecast_df.loc[len(forecast_df.index) - 1, ['date', param_name]] = [date,
                                                                                                             value]
                                        forecast_df.sort_values('date')
                                    else:
                                        forecast_df.loc[forecast_df['date'] == date, param_name] = value
                                    os.remove(csv_file_path)

                            break  # check only first-level subdirectories
                    break  # check only first-level subdirectories

                offset = offset + 3
            forecast_df.to_csv(final_csv_path)
        init_date = init_date + datetime.timedelta(days=1)


def process_csv_files():
    print("Processing csv files...")
    for root, dirs, filenames in os.walk("download/csv"):
        for dir_with_csvs in dirs:
            latlon_search = re.search(r'(\d+(_\d)?)-(\d+(_\d)?)', dir_with_csvs)
            latitude = latlon_search.group(1)
            longitude = latlon_search.group(3)
            prepare_final_csvs_from_csvs(dir_with_csvs, latitude, longitude)
        break


def process_netCDF_file_to_csv(file):
    pass


def process_netCDF_files():
    print("Processing netCDF files...")
    netCDF_dir = "download/netCDF"
    files_in_directory = [f for f in os.listdir(netCDF_dir) if isfile(join(netCDF_dir, f)) and f.endswith(".nc")]
    for file in files_in_directory:
        process_netCDF_file_to_csv(file)


if __name__ == '__main__':
    process_csv_files()
    process_netCDF_files()

    # path = '../gfs_processor/download/netCDF/gfs.0p25.2015011500.f009.grib2.belniak.nc'
    # ds = nc.Dataset(path)
    #
    # print(ds.dimensions['ncl_strlen_0']):


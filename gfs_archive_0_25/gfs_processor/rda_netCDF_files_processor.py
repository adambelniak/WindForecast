import time
from os.path import isfile, join
import netCDF4 as nc
import os
import re
import datetime
import sys
import pandas as pd
import schedule
import tqdm

sys.path.insert(1, '../..')

from gfs_archive_0_25.gfs_processor.own_logger import logger
from gfs_archive_0_25.utils import prep_zeros_if_needed
from gfs_archive_0_25.gfs_processor.consts import *


def get_forecast_df_for_date_and_run(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=[0])
    else:
        return pd.DataFrame(columns=["date"])


def get_date_run_offset_from_netCDF_file_name(file):
    date_matcher = re.match(RAW_NETCDF_FILENAME_REGEX, file)
    date_from_filename = date_matcher.group(2)
    year = date_from_filename[:4]
    month = date_from_filename[4:6]
    day = date_from_filename[6:8]
    return datetime.datetime(int(year), int(month), int(day)), date_from_filename[8:10], date_matcher.group(3)


def get_value_from_netCDF(dataset, lat_index, lon_index):
    return next(iter(dataset.variables.values()))[0][lat_index][lon_index]


def process_netCDF_file_to_csv(file, param, level):
    ds = nc.Dataset(os.path.join("download", "netCDF", param, level, file))
    lats, lons = ds.variables['lat_0'][:], ds.variables['lon_0'][:]
    init_date, run, offset = get_date_run_offset_from_netCDF_file_name(file)
    date_str = (init_date + datetime.timedelta(hours=int(offset))).isoformat()

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


def process_netCDF_files():
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


def schedule_processing():
    try:
        job = schedule.every(15).minutes.do(lambda: process_netCDF_files())
        job.run()
        while True:
            schedule.run_pending()
            time.sleep(60)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    schedule_processing()




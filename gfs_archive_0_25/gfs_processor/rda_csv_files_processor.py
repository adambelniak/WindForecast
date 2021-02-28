import glob
import os
import re
import datetime
import sys
import pandas as pd

sys.path.insert(1, '../..')

from gfs_archive_0_25.gfs_processor.own_logger import logger
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


def get_forecast_df_for_date_and_run(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=[0])
    else:
        return pd.DataFrame(columns=["date"])


def prepare_final_csvs_from_csvs(dir_with_csvs, latitude: str, longitude: str, init_date):
    final_date = datetime.datetime.now()
    date = init_date
    latlon_dir = os.path.join("download/csv", dir_with_csvs)
    final_csv_dir = os.path.join("output_csvs", latitude.replace('.', '_') + '-' + longitude.replace('.', '_'))

    if not os.path.exists(final_csv_dir):
        os.makedirs(final_csv_dir)

    while date < final_date:
        for run in ['00', '06', '12', '18']:
            # final csv name for this forecast
            final_csv_name = FINAL_CSV_FILENAME_FORMAT.format(date.year,
                                                              prep_zeros_if_needed(str(date.month), 1),
                                                              prep_zeros_if_needed(str(date.day), 1),
                                                              run)
            final_csv_path = os.path.join(final_csv_dir, final_csv_name)
            forecast_df = get_forecast_df_for_date_and_run(final_csv_path)

            offset = 3
            while offset < 168:
                # fetch values from all params and levels
                for root, param_dirs, filenames in os.walk(latlon_dir):
                    for param_dir in param_dirs:
                        for root, level_dirs, filenames in os.walk(os.path.join(latlon_dir, param_dir)):
                            for level_dir in level_dirs:
                                level_dir_path = os.path.join(latlon_dir, param_dir, level_dir)
                                csv_file_glob = RDA_CSV_FILENAME_FORMAT.format(date.year,
                                                                               prep_zeros_if_needed(str(date.month), 1),
                                                                               prep_zeros_if_needed(str(date.day), 1),
                                                                               run,
                                                                               prep_zeros_if_needed(str(offset), 2),
                                                                               '*')
                                csv_file_path = os.path.join(level_dir_path, csv_file_glob)
                                csv_files_found = glob.glob(csv_file_path)

                                if len(csv_files_found) != 0:
                                    csv_file = csv_files_found[0]
                                    value = get_value_from_csv(csv_file)
                                    param_name = param_dir + '_' + level_dir.replace("_", ":").replace(",", "_")
                                    date = (date + datetime.timedelta(hours=offset)).isoformat()

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
                                    #os.remove(csv_file_path)

                            break  # check only first-level subdirectories
                    break  # check only first-level subdirectories

                offset = offset + 3
            forecast_df.to_csv(final_csv_path)
        date = date + datetime.timedelta(days=1)


def process_csv_files():
    logger.info("Processing csv files...")
    for root, dirs, filenames in os.walk("download/csv"):
        for dir_with_csvs in dirs:
            latlon_search = re.search(r'(\d+(_\d)?)-(\d+(_\d)?)', dir_with_csvs)
            latitude = latlon_search.group(1)
            longitude = latlon_search.group(3)
            prepare_final_csvs_from_csvs(dir_with_csvs, latitude, longitude, datetime.datetime(2015, 1, 15))
        break


if __name__ == '__main__':
    process_csv_files()
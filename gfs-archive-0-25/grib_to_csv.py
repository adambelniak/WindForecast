#!/usr/bin/env python
import os
import csv
import re
import datetime
import argparse
from os.path import isfile, join

from common_grib import fetch_data_from_grib


grib_filename_pattern = 'gfs.0p25.(\d{10}).f(\d{3}).grib2.*'


def save_to_csv(data, out_filepath, field_names):
    out_file = open(out_filepath, 'a')
    with out_file:
        writer = csv.DictWriter(out_file, fieldnames=field_names)
        writer.writerow(data)
        out_file.close()


def extract_data_to_csv(filepath, output_dir, latitude, longitude, gfs_parameter):
    raw_date = re.search(grib_filename_pattern, filepath).group(1)
    offset = re.search(grib_filename_pattern, filepath).group(2)
    run = raw_date[8:10]
    init_date = '{0}-{1}-{2}'.format(raw_date[0:4], raw_date[4:6], raw_date[6:8])
    date = datetime.datetime(int(raw_date[0:4]), int(raw_date[4:6]), int(raw_date[6:8]))
    date = date.replace(hour=int(run)) + datetime.timedelta(hours=int(offset))

    data = {
        'date': date.isoformat(),
        gfs_parameter['fullName']: fetch_data_from_grib(filepath, [gfs_parameter], latitude, longitude)[gfs_parameter['fullName']]
    }

    csv_filename = init_date + '-' + run + 'Z.csv'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    csv_out_path = os.path.join(output_dir, csv_filename)

    save_to_csv(data, csv_out_path, ['date', gfs_parameter['fullName']])


def grib_to_csv(input_dir, output_dir, latitude, longitude, shortName, typeOfLevel, level):
    files_in_directory = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f)) and bool(re.search(grib_filename_pattern, f))]

    print("Found " + str(len(files_in_directory)) + " matching files.")
    files_in_directory.sort(key=lambda str: re.search(grib_filename_pattern, str).group(2))
    gfs_parameter = {
        "shortName": shortName,
        "fullName": shortName
    }
    if typeOfLevel is not None:
        gfs_parameter['typeOfLevel'] = typeOfLevel
        gfs_parameter['level'] = level

    for f in files_in_directory:
        extract_data_to_csv(os.path.join(input_dir, f), output_dir, latitude, longitude, gfs_parameter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', help='Directory with grib files. Default: current directory.', default='')
    parser.add_argument('--out', help='Output directory for csv files. Default: current directory.', default='')
    parser.add_argument('--lat', help='Latitude of point for which to get the data. Default: 54.6', default=54.6, type=float)
    parser.add_argument('--long', help='Longitude of point for which to get the data. Default: 18.8', default=18.8, type=float)
    parser.add_argument('--shortName', help='Short name of parameter to fetch. Default: 2t', default='2t')
    parser.add_argument('--type', help='Type of level')
    parser.add_argument('--level', help='Level', type=int)

    args = parser.parse_args()
    if ('type' in vars(args) and 'level' not in vars(args) or
            'level' in vars(args) and 'type' not in vars(args)):
        parser.error('--type and --level cannot be specified separately.')

    grib_to_csv(args.dir, args.out, args.lat, args.long, args.shortName, args.type, args.level)

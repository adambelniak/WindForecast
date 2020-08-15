#!/usr/bin/env python
import os
import csv
import re
import argparse
import pygrib
import gc
from os.path import isfile, join

from utils import prep_zeros_if_needed
from common_grib import fetch_data_from_grib

gfs_variables = [
    {"shortName": "gust",
     "fullName": "Gusts"},
    {"shortName": "tcc",
     "fullName": "400TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 400},
    {"shortName": "tcc",
     "fullName": "450TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 450},
    {"shortName": "tcc",
     "fullName": "500TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 500},
    {"shortName": "tcc",
     "fullName": "550TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 550},
    {"shortName": "tcc",
     "fullName": "400TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 600},
    {"shortName": "tcc",
     "fullName": "400TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 660},
    {"shortName": "tcc",
     "fullName": "700TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 700},
    {"shortName": "tcc",
     "fullName": "800TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 800},
    {"shortName": "tcc",
     "fullName": "850TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 850},
    {"shortName": "tcc",
     "fullName": "900TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 900},
    {"shortName": "tcc",
     "fullName": "950TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 950},
    {"shortName": "tcc",
     "fullName": "1000TCC",
     "typeOfLevel": "isobaricInhPa",
     "level": 1000},
    {"shortName": "t",
     "fullName": "SurfaceT",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "2t",
     "fullName": "2mT",
     "typeOfLevel": "heightAboveGround",
     "level": 2},
    {"shortName": "10u",
     "fullName": "10m-uwind"},
    {"shortName": "10v",
     "fullName": "10m-vwind"},
    {"shortName": "cprat",
     "fullName": "ConvectivePrate",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "prate",
     "fullName": "Prate",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "prmsl",
     "fullName": "Pressure"}
]

grib_filename_pattern = 'gfs.0p25.(\d{10}).f(\d{3}).grib2.*'


def save_to_csv(data, out_filepath, field_names):
    out_file = open(out_filepath, 'a')
    with out_file:
        writer = csv.DictWriter(out_file, fieldnames=field_names)
        writer.writerow(data)
        out_file.close()


def check_if_variable_exists(variable, grib):
    try:
        if 'typeOfLevel' in variable:
            message = grib.select(shortName=variable['shortName'], typeOfLevel=variable['typeOfLevel'],
                                  level=variable['level'])[0]
        else:
            message = grib.select(shortName=variable['shortName'])[0]
        return True
    except:
        return False


def extract_data_to_csv(filepath, output_dir, latitude, longitude):
    grib = pygrib.open(filepath)
    full_names = ['date']
    data = {}
    for variable in gfs_variables:
        if check_if_variable_exists(variable, grib):
            data[variable['fullName']] = fetch_data_from_grib(filepath, [variable], latitude, longitude)[
                variable['fullName']]
            full_names.append(variable['fullName'])

    grib.close()
    raw_date = re.search(grib_filename_pattern, filepath).group(1)
    offset = re.search(grib_filename_pattern, filepath).group(2)

    date = '{0}-{1}-{2}'.format(raw_date[0:4], raw_date[4:6], raw_date[6:8])
    run = raw_date[8:10]
    csv_filename = date + '-' + run + 'Z.csv'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    csv_out_path = os.path.join(output_dir, csv_filename)
    data['date'] = date + ' ' + prep_zeros_if_needed(str((int(run) + int(offset)) % 24), 1) + ':00'
    save_to_csv(data, csv_out_path, full_names)


def grib_to_csv(input_dir, output_dir, latitude=0., longitude=0.):
    files_in_directory = [f for f in os.listdir(input_dir) if isfile(join(input_dir, f)) and bool(re.search(grib_filename_pattern, f))]

    print("Found " + str(len(files_in_directory)) + " matching files.")
    files_in_directory.sort(key=lambda str: re.search(grib_filename_pattern, str).group(2))

    for f in files_in_directory:
        extract_data_to_csv(os.path.join(input_dir, f), output_dir, latitude, longitude)
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', help='Directory with grib files', default='')
    parser.add_argument('--out', help='Output directory for csv files', default='')
    parser.add_argument('--lat', help='Latitude of point for which to get the data', default=54.6, type=float)
    parser.add_argument('--long', help='Longitude of point for which to get the data', default=18.8, type=float)

    args = parser.parse_args()
    grib_to_csv(args.dir, args.out, args.lat, args.long)

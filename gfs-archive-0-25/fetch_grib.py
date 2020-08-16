#!/usr/bin/env python
import calendar
import os
import sys
import csv
import requests
import time

from past.builtins import raw_input
from utils import prep_zeros_if_needed
from common_grib import fetch_data_from_grib


gfs_parameters = [
    {"shortName": "refc",
     "fullName": "Maximum/Composite radar reflectivity, dBz"},
    {"shortName": "gust",
     "fullName": "Wind speed (gust), m/s"},
    {"shortName": "tcc",
     "fullName": "400 Total Cloud Cover, %",
     "typeOfLevel": "isobaricInhPa",
     "level": 400},
    {"shortName": "tcc",
     "fullName": "600 Total Cloud Cover, %",
     "typeOfLevel": "isobaricInhPa",
     "level": 600},
    {"shortName": "tcc",
     "fullName": "800 Total Cloud Cover, %",
     "typeOfLevel": "isobaricInhPa",
     "level": 800},
    {"shortName": "tcc",
     "fullName": "950 Total Cloud Cover, %",
     "typeOfLevel": "isobaricInhPa",
     "level": 950},
    {"shortName": "tcc",
     "fullName": "1000 Total Cloud Cover, %",
     "typeOfLevel": "isobaricInhPa",
     "level": 1000},
    {"shortName": "t",
     "fullName": "Surface temperature, K",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "2t",
     "fullName": "2m temperature, K",
     "typeOfLevel": "heightAboveGround",
     "level": 2},
    {"shortName": "2r",
     "fullName": "2m humidity, %",
     "typeOfLevel": "heightAboveGround",
     "level": 2},
    {"shortName": "10u",
     "fullName": "10 metre U wind component, m/s"},
    {"shortName": "10v",
     "fullName": "10 metre V wind component, m/s"},
    {"shortName": "cprat",
     "fullName": "Convective precipitation rate, mm/h",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "prate",
     "fullName": "Precipitation rate, mm/h",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "prmsl",
     "fullName": "Pressure reduced to MSL, Pa"}
]
hel_long = 18.8
hel_lat = 54.6
dspath = 'https://rda.ucar.edu/data/ds084.1/'
grib_filename_template = '{0}/{0}{1}{2}/gfs.0p25.{0}{1}{2}{3}.f{4}.grib2'
csv_filename_template = '{0}-{1}-{2}-{3}Z.csv'


def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size / filesize) * 100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()


def authenticate_to_rda():
    # Try to get password
    try:
        import getpass
        input = getpass.getpass
    except:
        try:
            input = raw_input
        except:
            pass
    pswd = input('Password to rda: ')

    url = 'https://rda.ucar.edu/cgi-bin/login'
    values = {'email': 'belniakm@wp.pl', 'passwd': pswd, 'action': 'login'}
    # Authenticate
    ret = requests.post(url, data=values)
    if ret.status_code != 200:
        print('Bad Authentication')
        print(ret.text)
        exit(1)
    return ret


def download_file(file, cookies):
    filename = dspath + file
    file_base = os.path.basename(file)
    print('Downloading', file_base)
    req = requests.get(filename, cookies=cookies, allow_redirects=True, stream=True)
    filesize = int(req.headers['Content-length'])
    with open(file_base, 'wb') as outfile:
        chunk_size = 1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(file_base, filesize)
    check_file_status(file_base, filesize)
    print()


def save_test_data():
    year = '2020'
    month = '07'
    day = '30'
    run = '00'
    hour = '015'
    filename = grib_filename_template.format(year, month, day, run, hour)
    if os.path.exists(filename.split('/')[-1]) is False:
        cookie_with_auth = authenticate_to_rda().cookies
        download_file(filename, cookie_with_auth)
    out_filepath = csv_filename_template.format(year, month, day, run)
    out_file = open(out_filepath, 'w')
    with out_file:
        full_names = []
        for parameter in gfs_parameters:
            full_names.append(parameter['fullName'])
        writer = csv.DictWriter(out_file, fieldnames=full_names)
        writer.writeheader()
        data = fetch_data_from_grib(filename.split('/')[-1], gfs_parameters, hel_lat, hel_long)
        print("Saving data from file " + filename + " to file " + out_filepath)
        writer.writerow(data)
        out_file.close()
        print("Data saved.")


def save_all_data():
    year = 2020
    gfs_runs = ['00', '06', '12', '18']
    cookie_with_auth = authenticate_to_rda().cookies
    for month in range(13):
        for day in range(calendar.monthrange(year, month + 1)[1]):  # January is '1'
            for run in gfs_runs:
                out_filepath = csv_filename_template.format(str(year),
                                                            prep_zeros_if_needed(str(month + 1), 1),
                                                            prep_zeros_if_needed(str(day + 1), 1),
                                                            run)
                try:
                    os.remove(out_filepath)
                except OSError:
                    pass
                out_file = open(out_filepath, 'a')
                with out_file:
                    full_names = ['offset']
                    for parameter in gfs_parameters:
                        full_names.append(parameter['fullName'])
                    writer = csv.DictWriter(out_file, fieldnames=full_names)
                    writer.writeheader()
                    for hour in range(0, 168, 3):
                        filename = grib_filename_template.format(year,
                                                                 prep_zeros_if_needed(str(month + 1), 1),
                                                                 prep_zeros_if_needed(str(day + 1), 1),
                                                                 run,
                                                                 prep_zeros_if_needed(str(hour), 2))
                        filename_base = filename.split('/')[-1]
                        if os.path.exists(filename_base) is False:
                            download_file(filename, cookie_with_auth)
                        data = fetch_data_from_grib(filename_base, gfs_parameters, hel_lat, hel_long)
                        data['offset'] = prep_zeros_if_needed(str(hour), 2)
                        os.remove(filename_base)
                        print("Saving data from file: " + filename_base + " to file " + out_filepath)
                        writer.writerow(data)
    print("All data saved.")


if __name__ == '__main__':
   if len(sys.argv) > 1 and sys.argv[1] == 'prod':
        start = time.time()
        save_all_data()
        end = time.time()
        print(end - start)
    else:
        save_test_data()
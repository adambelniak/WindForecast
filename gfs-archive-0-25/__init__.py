#!/usr/bin/env python

# import argparse
import calendar
import os
import sys
import csv
import pygrib
import requests

from past.builtins import raw_input
from utils import get_nearest_coords
from utils import prep_zeros_if_needed
from scipy import interpolate

variables_names = [
    {"shortName": "refc",
     "fullName": "Maximum/Composite radar reflectivity"},
    {"shortName": "gust",
     "fullName": "Wind speed (gust)"},
    {"shortName": "tcc",
     "fullName": "400 Total Cloud Cover",
     "typeOfLevel": "isobaricInhPa",
     "level": 400},
    {"shortName": "tcc",
     "fullName": "600 Total Cloud Cover",
     "typeOfLevel": "isobaricInhPa",
     "level": 600},
    {"shortName": "tcc",
     "fullName": "800 Total Cloud Cover",
     "typeOfLevel": "isobaricInhPa",
     "level": 800},
    {"shortName": "tcc",
     "fullName": "950 Total Cloud Cover",
     "typeOfLevel": "isobaricInhPa",
     "level": 950},
    {"shortName": "tcc",
     "fullName": "1000 Total Cloud Cover",
     "typeOfLevel": "isobaricInhPa",
     "level": 1000},
    {"shortName": "t",
     "fullName": "Surface temperature",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "2t",
     "fullName": "2m temperature",
     "typeOfLevel": "heightAboveGround",
     "level": 2},
    {"shortName": "2r",
     "fullName": "2m humidity",
     "typeOfLevel": "heightAboveGround",
     "level": 2},
    {"shortName": "10u",
     "fullName": "10 metre U wind component"},
    {"shortName": "10v",
     "fullName": "10 metre V wind component"},
    {"shortName": "cprat",
     "fullName": "Convective precipitation rate",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "prate",
     "fullName": "Precipitation rate",
     "typeOfLevel": "surface",
     "level": 0},
    {"shortName": "prmsl",
     "fullName": "Pressure reduced to MSL"}
]
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
    if len(sys.argv) < 3 and not 'RDAPSWD' in os.environ:
        try:
            import getpass
            input = getpass.getpass
        except:
            try:
                input = raw_input
            except:
                pass
        pswd = input('Password: ')
    else:
        try:
            pswd = sys.argv[1]
        except:
            pswd = os.environ['RDAPSWD']

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


def fetch_gfs_archive_data(grib_file, latitude=0., longitude=0.):
    print("Fetching gfs data from file " + grib_file)
    gr = pygrib.open(grib_file)
    data = []

    for variable in variables_names:
        if 'typeOfLevel' in variable:
            message = gr.select(shortName=variable['shortName'], typeOfLevel=variable['typeOfLevel'], level=variable['level'])[0]
        else:
            message = gr.select(shortName=variable['shortName'])[0]

        coords = get_nearest_coords(latitude, longitude)
        values = message.data(lat1=coords[0][0],
                            lat2=coords[0][1],
                            lon1=coords[1][0],
                            lon2=coords[1][1])

        # pygrib needs to have latitude from smaller to larger, but returns the values north-south,
        # so from larger lat to smaller lat :( Thus, reversing 'data' array
        interpolated_function_for_values = interpolate.interp2d(coords[0], coords[1], values[0][::-1])
        data.append(interpolated_function_for_values(latitude, longitude).item())

    return data

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Fetch gsf 0.25° archive forecasts and save to CSV.')
    # TODO argument parsing - from year to year, which run (00, 06, 12 or 18UTC) itp...

    if len(sys.argv) > 1 and sys.argv[1] == 'prod':
        year = 2020
        gfs_runs = ['00', '06', '12', '18']
        cookie_with_auth = authenticate_to_rda().cookies
        for month in range(12):
            for day in range(calendar.monthrange(year, month + 1)[1]):  # January is '1'
                for run in gfs_runs:
                    out_filename = csv_filename_template.format(str(year),
                                                                prep_zeros_if_needed(str(month + 1), 1),
                                                                prep_zeros_if_needed(str(day + 1), 1),
                                                                run)
                    try:
                        os.remove(out_filename)
                    except OSError:
                        pass
                    out_file = open(out_filename, 'a')
                    with out_file:
                        writer = csv.writer(out_file)
                        fullNames = []
                        for variable in variables_names:
                            fullNames.append(variable['fullName'])
                        writer.writerow(fullNames)

                        for hour in range(0, 240, 3):
                            filename = grib_filename_template.format(year,
                                                                     prep_zeros_if_needed(str(month + 1), 1),
                                                                     prep_zeros_if_needed(str(day + 1), 1),
                                                                     run,
                                                                     prep_zeros_if_needed(str(hour), 2))
                            download_file(filename, cookie_with_auth)
                            data = fetch_gfs_archive_data(filename.split('/')[-1], 54.6, 18.8)
                            print("Saving data from file: " + filename + " to file " + out_filename)
                            data.insert(0, prep_zeros_if_needed(str(hour), 2))
                            writer.writerow(data)  # coords for Hel
    else:
        year = '2020'
        month = '07'
        day = '30'
        run = '00'
        hour = '015'
        filename = grib_filename_template.format(year, month, day, run, hour)
        out_filename = csv_filename_template.format(year, month, day, run)
        out_file = open(out_filename, 'w')
        with out_file:
            writer = csv.writer(out_file)
            fullNames = []
            for variable in variables_names:
                fullNames.append(variable['fullName'])
            writer.writerow(fullNames)

            data = fetch_gfs_archive_data(filename.split('/')[-1], 54.6, 18.8)
            print("Saving data from file " + filename + " to file " + out_filename)
            writer.writerow(data)  # coords for Hel

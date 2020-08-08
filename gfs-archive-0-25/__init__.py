#!/usr/bin/env python

# import argparse
import calendar
import os
import sys

import pygrib
import requests

from past.builtins import raw_input
import utils
from utils import prep_zeros_if_needed
from scipy import interpolate

dspath = 'https://rda.ucar.edu/data/ds084.1/'
filename_template = '{0}/{0}{1}{2}/gfs.0p25.{0}{1}{2}{3}.f{4}.grib2'


def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size / filesize) * 100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()


def authenticate_to_rda():
    # Try to get password
    if len(sys.argv) < 2 and not 'RDAPSWD' in os.environ:
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


def save_GFS_archive_data(grib_file, latitude=0., longitude=0.):
    gr = pygrib.open(grib_file.split('/')[-1])
    u_component_of_wind_msgs = gr.select(shortName='10u')[0]
    v_component_of_wind_msgs = gr.select(shortName='10v')[0]

    # for g in gr:
    #     print(g.typeOfLevel, g.level, g.name, g.shortName, g.validDate, g.analDate, g.forecastTime)

    coords = utils.get_nearest_coords(latitude, longitude)
    print("Getting grid data for coords: " + str(coords))
    data_u = u_component_of_wind_msgs.data(lat1=coords[0][0],
                                           lat2=coords[0][1],
                                           lon1=coords[1][0],
                                           lon2=coords[1][1])

    data_v = v_component_of_wind_msgs.data(lat1=coords[0][0],
                                           lat2=coords[0][1],
                                           lon1=coords[1][0],
                                           lon2=coords[1][1])

    # pygrib needs to have latitude from smaller to larger, but returns the values north-south,
    # so from larger lat to smaller lat :( Thus, reversing 'data' array
    interpolated_function_u = interpolate.interp2d(coords[0], coords[1], data_u[0][::-1])
    interpolated_function_v = interpolate.interp2d(coords[0], coords[1], data_v[0][::-1])
    print("Value of u-wind component at 20m for point (" + str(latitude) + ", " + str(longitude) + "): " +
          str(interpolated_function_u(latitude, longitude)))

    print("Value of v-wind component at 20m for point (" + str(latitude) + ", " + str(longitude) + "): " +
          str(interpolated_function_v(latitude, longitude)))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Fetch gsf 0.25° archive forecasts and save to CSV.')
    # TODO argument parsing - from year to year, which run (00, 06, 12 or 18UTC) itp...

    filename = filename_template.format(
        '2020',
        '07',
        '30',
        '00',
        '015'
    )

    # cookie_with_auth = authenticate_to_rda().cookies
    # year = '2020'
    # gfs_runs = ['00', '06', '12', '18']
    # for month in range(12):  # January is '1'
    #     for day in range(calendar.monthrange(year, month + 1)):
    #         for run in gfs_runs:
    #             for hour in range(0, 240, 3):
    #                 download_file(filename_template.format(
    #                     year,
    #                     prep_zeros_if_needed(str(month), 1),
    #                     prep_zeros_if_needed(str(day), 1),
    #                     run,
    #                     prep_zeros_if_needed(str(hour), 2)
    #                 ), cookie_with_auth)

    # download_file(filename, cookie_with_auth)
    save_GFS_archive_data(filename, 54.6, 18.8) # coords for Hel

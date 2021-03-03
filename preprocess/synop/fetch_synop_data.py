import glob

from bs4 import BeautifulSoup
import requests
import re
from pathlib import Path
import os
from zipfile import ZipFile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import argparse

import preprocess.synop.consts as consts

"""This code obtains SYNOP data from 'https://danepubliczne.imgw.pl/'.

    SYNOP file containing meteorological synoptic data for single localisation and year. 

    https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/s_t_format.txt 
    - under this url the description of file format is available. For project purpose it's required to process at least three columns, 
    which contain wind direction, wind velocity and gust of wind available at columns:
     - direction - 23
     - velocity - 25
     - gust - 27 
    and date columns:
     - year - 2
     - month - 3
     - day - 4
     - hour - 5
     Also, there are more interesting attributes that are planned to be analyzed in the future:
     - 52 - current weather
     - 29 - temperature at 2 meters
     - 41 - pressure
"""

def download_list_of_station(dir: str):
    file_name = 'wykaz_stacji.csv'

    if not os.path.isfile(os.path.join(dir, file_name)):
        url = 'https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/' + file_name
        file = requests.get(url, stream=True)
        opened_file = open(os.path.join(dir, file_name), 'wb')
        opened_file.write(file.content)
        opened_file.close()


def get_localisation_id(localisation_name: str, code_fallback, dir='synop_data'):
    loc_data = pd.read_csv(os.path.join(dir, 'wykaz_stacji.csv'), encoding="latin-1",
                           names=['unknown', 'city_name', 'meteo_code'],  dtype='str')
    row = loc_data.loc[loc_data['city_name'] == localisation_name]

    if row.shape[0] == 0:
        if code_fallback is not None:
            row = loc_data.loc[loc_data['meteo_code'] == code_fallback]
            if row.shape[0] == 0:
                raise Exception("Location does not exists")
        else:
            raise Exception("Location does not exists")

    return row.iloc[0]['meteo_code'], row.iloc[0]['city_name']


def get_synop_data(localisation_code: str, year: str, dir: str):
    dir_per_year = os.path.join(dir, year, 'download')
    Path(dir_per_year).mkdir(parents=True, exist_ok=True)

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/" + year

    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    zip_rows = soup.find_all('tr')

    for row in zip_rows:
        td_link = row.find('a')
        if td_link:
            contains_zip_file = re.match(rf'^(.+?){localisation_code}(.+?).zip$', td_link['href'])
            if contains_zip_file:
                file = requests.get(url + '/' + td_link['href'], stream=True)
                opened_file = open(os.path.join(dir_per_year, td_link['href']), 'wb')
                opened_file.write(file.content)
                opened_file.close()


def extract_zip_files(year: str, dir: str):
    dir_with_zip = os.path.join(dir, year, 'download')
    data_directory = os.path.join(dir, year)

    for file in os.listdir(dir_with_zip):
        with ZipFile(os.path.join(dir_with_zip, file), 'r') as zip:
            zip.extractall(path=data_directory)


def read_data(localisation_code: str, year: str, columns, dir='synop_data'):
    station_data = pd.DataFrame(columns=list(list(zip(*columns))[1]))

    for filepath in glob.iglob(rf'{dir}/{year}/*{localisation_code}*.csv', recursive=True):
        synop_data = pd.read_csv(filepath, encoding="ISO-8859-1", header=None)
        required_data = synop_data[list(list(zip(*columns))[0])]
        station_data[list(list(zip(*columns))[1])] = required_data
    return station_data


def plot_scatter_data_for_year(localisation_code: str, localisation_name:str, year: int, dir='synop_data'):
    station_data = pd.read_csv(os.path.join(dir, f"{localisation_name}_{localisation_code}_data.csv"))
    one_year_data = station_data.loc[station_data['year'] == year]
    one_year_data['date_time'] = pd.to_datetime(one_year_data[['year', 'month', 'day', 'hour']])

    one_year_data.plot(kind='scatter', x='date_time', y=consts.VELOCITY_COLUMN[1], color='red')
    plt.show()


def plot_each_month_in_year(localisation_code: str, localisation_name: str, year: int, dir='synop_data'):
    station_data = pd.read_csv(os.path.join(dir, f"{localisation_name}_{localisation_code}_data.csv"))
    one_year_data = station_data.loc[station_data['year'] == year]
    one_year_data['date_time'] = pd.to_datetime(one_year_data[['year', 'month', 'day', 'hour']])

    sns.boxplot(x='month', y="pressure",
                data=one_year_data, palette="Set3")

    plt.show()


def plot_box_all_data(localisation_code: str, localisation_name: str, dir='synop_data'):
    station_data = pd.read_csv(os.path.join(dir, f"{localisation_name}_{localisation_code}_data.csv"))
    print(station_data.head())
    sns.boxplot(x='year', y=consts.VELOCITY_COLUMN[1],
                data=station_data, palette="Set3")

    plt.show()


def process_all_data(from_year, until_year, localisation_code, localisation_name, dir='synop_data'):
    columns = [consts.YEAR, consts.MONTH, consts.DAY, consts.HOUR, consts.DIRECTION_COLUMN, consts.VELOCITY_COLUMN,
               consts.GUST_COLUMN, consts.TEMPERATURE, consts.PRESSURE, consts.CURRENT_WEATHER]
    station_data = pd.DataFrame(columns=list(list(zip(*columns))[1]))
    localisation_code = localisation_code.strip()

    for year in tqdm.tqdm(range(from_year, until_year)):
        get_synop_data(localisation_code, str(year), dir)
        extract_zip_files(str(year), dir)
        processed_data = read_data(localisation_code, str(year), columns, dir)
        station_data = station_data.append(processed_data)
    station_data.to_csv(os.path.join(dir, f"{localisation_name}_{localisation_code}_data.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', help='Working directory', default='')
    parser.add_argument('--out', help='Directory where to save synop files', default='synop_data')
    parser.add_argument('--localisation_name', help='Localisation name for which to get data', default=None, type=str)
    parser.add_argument('--code_fallback', help='Localisation code as a fallback if name is not found', default=None, type=str)
    parser.add_argument('--start_year', help='Start date for fetching data', type=int, default=2001)
    parser.add_argument('--end_year', help='End date for fetching data', type=int, default=2021)

    parser.add_argument('--plot_box_year', help='Year fow which create box plot for each month', type=int, default=2019)


    args = parser.parse_args()
    if args.localisation_name is None and args.code_fallback is None:
        raise Exception("Please provide either localisation_name or code_fallback!")
    download_list_of_station(args.dir)
    localisation_code, name = get_localisation_id(args.localisation_name, args.code_fallback, args.dir)
    localisation_code = localisation_code.strip()
    localisation_name = args.localisation_name
    if localisation_name is None:
        localisation_name = name
    process_all_data(args.start_year, args.end_year, str(localisation_code), localisation_name)
    plot_scatter_data_for_year(str(localisation_code), localisation_name, args.plot_box_year, args.out)

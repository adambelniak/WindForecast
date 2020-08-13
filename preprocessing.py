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

"""This code obtains SYNOP data from 'https://danepubliczne.imgw.pl/'.
    
    SYNOP file contains meteorological data for single localisation and year. 
    
    https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/s_t_format.txt 
    - under this url, is available descriptions of file format. For project purpose it's required to process at least three columns, 
    which contains wind direction, wind velocity and gust of wind which are available at column:
     - direction - 23
     - velocity - 25
     - gust - 27 
    and date columns:
     - year - 2
     - month - 3
     - day - 4
     - hour - 5
"""
YEAR = 2
MONTH = 3
DAY = 4
HOUR = 5
DIRECTION_COLUMN = 23
VELOCITY_COLUMN = 25
GUST_COLUMN = 27


def download_list_of_station(dir: str):
    file_name = 'wykaz_stacji.csv'
    url = 'https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/' + file_name
    file = requests.get(url, stream=True)
    opened_file = open(os.path.join(dir, file_name), 'wb')
    opened_file.write(file.content)
    opened_file.close()


def get_localisation_id(localisation_name: str, dir='synop_data'):
    loc_data = pd.read_csv(os.path.join(dir, 'wykaz_stacji.csv'), encoding="ISO-8859-1",
                           names=['unknown', 'city_name', 'meteo_code'])
    row = loc_data.loc[loc_data['city_name'] == localisation_name]

    if row.shape[0] == 0:
        raise Exception("Location does not exists")
    return row.iloc[0]['meteo_code']


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


def read_data(localisation_code: str, year: str, number_column, columns, dir='synop_data'):
    station_data = pd.DataFrame(columns=columns)

    for filepath in glob.iglob(rf'{dir}/{year}/*{localisation_code}*.csv', recursive=True):
        synop_data = pd.read_csv(filepath, encoding="ISO-8859-1", header=None)
        required_data = synop_data[number_column]
        station_data[columns] = required_data
    return station_data


def plot_scatter_data_for_year(localisation_code: str, year: int, dir='synop_data'):
    station_data = pd.read_csv(os.path.join(dir, localisation_code + '_data.csv'))
    one_year_data = station_data.loc[station_data['year'] == year]
    one_year_data['date_time'] = pd.to_datetime(one_year_data[['year', 'month', 'day', 'hour']])

    one_year_data.plot(kind='scatter', x='date_time', y='velocity', color='red')
    plt.show()


def plot_each_month_in_year(localisation_code: str, year: int, dir='synop_data'):
    station_data = pd.read_csv(os.path.join(dir, localisation_code + '_data.csv'))
    one_year_data = station_data.loc[station_data['year'] == year]
    one_year_data['date_time'] = pd.to_datetime(one_year_data[['year', 'month', 'day', 'hour']])

    sns.boxplot(x='month', y="velocity",
                     data=one_year_data, palette="Set3")

    plt.show()


def process_all_data(from_year, until_year, localisation_code):
    dir = 'synop_data'
    columns = ['year', 'month', 'day', 'hour', 'direction', 'velocity', 'gust']
    station_data = pd.DataFrame(columns=columns)
    number_column = [YEAR, MONTH, DAY, HOUR, DIRECTION_COLUMN, VELOCITY_COLUMN, GUST_COLUMN]

    for year in tqdm.tqdm(range(from_year, until_year)):
        get_synop_data(localisation_code, str(year), dir)
        extract_zip_files(str(year), dir)
        processed_wind_data = read_data(localisation_code, str(year), number_column, columns, dir)
        station_data = station_data.append(processed_wind_data)
    station_data.to_csv(os.path.join(dir, localisation_code + '_data.csv'), index=False)


if __name__ == "__main__":
    process_all_data(2001, 2020, '135')
    plot_each_month_in_year('135', 2016)

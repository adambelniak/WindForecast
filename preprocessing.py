from bs4 import BeautifulSoup
import requests
import re
from pathlib import Path
import os
from zipfile import ZipFile
import pandas as pd


def download_list_of_station(dir: str):
    file_name = 'wykaz_stacji.csv'
    url = 'https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/' + file_name
    file = requests.get(url, stream=True)
    opened_file = open(os.path.join(dir, file_name), 'wb')
    opened_file.write(file.content)
    opened_file.close()


def get_localisation_id(localisation_name: str, dir='synop_data'):
    loc_data = pd.read_csv(os.path.join(dir, 'wykaz_stacji.csv'), encoding = "ISO-8859-1", names=['unknown', 'city_name', 'meteo_code'])
    row = loc_data.loc[loc_data['city_name'] == localisation_name]

    if row.shape[0] == 0:
        raise Exception("Location does not exists")
    return row.iloc[0]['meteo_code']


def get_synop_data(localisation_code, year: str, dir: str):
    dir_per_year = os.path.join(dir, year, 'download')
    Path(dir_per_year).mkdir(parents=True, exist_ok=True)

    url ="https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/" + year

    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    zip_rows = soup.find_all('tr')

    for row in zip_rows:
        td_link = row.find('a')
        if td_link:
            contains_zip_file = re.match(rf'^(.+?){localisation_code}(.+?).zip$', td_link['href'])
            if contains_zip_file:
                file = requests.get(url + '/' + td_link['href'],stream=True)
                opened_file = open(os.path.join(dir_per_year, td_link['href']), 'wb')
                opened_file.write(file.content)
                opened_file.close()


def extract_zip_files(year: str, dir: str):
    dir_with_zip = os.path.join(dir, year, 'download')
    data_directory = os.path.join(dir, year)

    for file in os.listdir(dir_with_zip):
        with ZipFile(os.path.join(dir_with_zip, file), 'r') as zip:
            zip.extractall(path=data_directory)


if __name__ == "__main__":
    dir = './synop_data'
    # extract_zip_files('2019', dir)
    # download_list_of_station(dir)
    localisation_code = get_localisation_id("HEL")
    get_synop_data(str(localisation_code), '2018', dir)
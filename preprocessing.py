from bs4 import BeautifulSoup
import requests
import re
from pathlib import Path
import os
from zipfile import ZipFile

def get_synop_data(year: str, dir: str):
    dir_per_year = os.path.join(dir, year, 'download')
    Path(dir_per_year).mkdir(parents=True, exist_ok=True)

    url ="https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/" + year

    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    zip_rows = soup.find_all('tr')

    for row in zip_rows[:5]:
        td_link = row.find('a')
        if td_link:
            contains_zip_file = re.match(r'^([a-zA-Z0-9\s_\\.\-\(\):])+\.zip$', td_link['href'])
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
    extract_zip_files('2019', dir)
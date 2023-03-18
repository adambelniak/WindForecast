import glob
import tarfile
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

import synop.consts as consts
from synop.tele_util import add_hourly_wind_velocity, add_hourly_direction, add_hourly_precipitation
from util.util import prep_zeros_if_needed

"""This code obtains SYNOP data from 'https://danepubliczne.imgw.pl/'.

    SYNOP file containing meteorological synoptic data for single localisation and year. 

    Wind data is obtained from https://dane.imgw.pl/datastore/.
    Wind and gusts velocity is obtained by taking an average from 7 observations from HH:30 to HH+1:30 
    Each file contains data for one month, one parameter for multiple localisations.

    https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/s_t_format.txt 
    - under this url the description of SYNOP file format is available. 
    Interesting parameters are located at the following columns:
     - direction - 23
     - velocity - 25
     - gust - 27 
     - year - 2
     - month - 3
     - day - 4
     - hour - 5
     - current weather - 53
     - temperature at 2 meters - 29
     - pressure - 41
"""

AUTO_STATION_CSV_COLUMNS = ['station_code', 'param_code', 'date', 'value']


def download_list_of_station(output_dir: str):
    file_name = 'wykaz_stacji.csv'

    out_file = os.path.join(output_dir, 'wykaz_stacji.csv')

    if not os.path.isfile(out_file):
        url = 'https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/' + file_name
        file = requests.get(url, stream=True)
        opened_file = open(out_file, 'wb')
        opened_file.write(file.content)
        opened_file.close()


def resolve_synop_localisation_name(station_code: str, station_list_dir: str):
    loc_data = pd.read_csv(os.path.join(station_list_dir, 'wykaz_stacji.csv'),
                           encoding="latin-1",
                           names=['meteo_code', 'station_name', 'short_meteo_code'], dtype='str')

    row = loc_data.loc[loc_data['meteo_code'] == station_code]
    if row.shape[0] == 0:
        raise Exception("Location does not exists")
    else:
        return row.iloc[0]['station_name']


def get_synop_data(localisation_code: str, year: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    url = "https://dane.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/" + year

    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    zip_rows = soup.find_all('tr')
    if len(localisation_code) > 3:
        localisation_code = localisation_code[-3:]

    for row in zip_rows:
        td_link = row.find('a')
        if td_link:
            contains_zip_file = re.match(rf'^(.+?){localisation_code}(.+?).zip$', td_link['href'])
            if contains_zip_file:
                file = requests.get(url + '/' + td_link['href'], stream=True)
                opened_file = open(os.path.join(output_dir, td_link['href']), 'wb')
                opened_file.write(file.content)
                opened_file.close()


def get_auto_station_data(year: str, month: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, f"Meteo_{year}-{month}.zip")
    if os.path.exists(output_file):
        return

    url = f"https://dane.imgw.pl/datastore/getfiledown/Arch/Telemetria/Meteo/{year}/Meteo_{year}-{month}.zip"
    if month in ['01', '02'] and year == '2021':
        url = url.replace('zip', 'ZIP')  # ¯\_(ツ)_/¯

    req = requests.get(url, stream=True)
    if req.status_code == 200:
        print(f"Downloading zip to {output_file}")
        with open(output_file, 'wb') as outfile:
            chunk_size = 1048576
            for chunk in req.iter_content(chunk_size=chunk_size):
                outfile.write(chunk)


def extract_tar_files(tar_dir: str, target_dir: str):
    for file in os.listdir(tar_dir):
        tar = tarfile.open(os.path.join(tar_dir, file))
        tar.extractall(target_dir)
        tar.close()
        os.remove(os.path.join(tar_dir, file))

def extract_zip_files(zip_dir: str, target_dir: str):
    for file in os.listdir(zip_dir):
        with ZipFile(os.path.join(zip_dir, file), 'r') as zip:
            zip.extractall(path=target_dir)
        os.remove(os.path.join(zip_dir, file))


def read_synop_data(columns: list, csv_location: str):
    station_data = pd.DataFrame(columns=list(list(zip(*columns))[1]))

    for filepath in glob.iglob(rf'{csv_location}/*.csv', recursive=True):
        synop_data = pd.read_csv(filepath, encoding="ISO-8859-1", header=None)
        required_data = synop_data[list(list(zip(*columns))[0])]
        station_data[list(list(zip(*columns))[1])] = required_data
    return station_data


def read_auto_station_data(csv_location: str, station_code: str):
    data = pd.DataFrame(columns=AUTO_STATION_CSV_COLUMNS)
    for filepath in glob.iglob(rf'{csv_location}/*.csv', recursive=True):
        station_data = pd.read_csv(filepath, encoding="ISO-8859-1", sep=';', index_col=False, dtype='str',
                                   names=AUTO_STATION_CSV_COLUMNS)
        required_data = station_data.loc[station_data['station_code'] == station_code]
        data = pd.concat([data, required_data])
    return data


def plot_scatter_data_for_year(localisation_code: str, localisation_name: str, year: int, dir='synop_data'):
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


def process_synop_data(from_year: int, until_year: int, localisation_code: str, localisation_name: str, working_dir: str):
    synop_data_dir = os.path.join(working_dir, 'synop_data')
    station_data = pd.DataFrame(columns=list(list(zip(*consts.SYNOP_FEATURES))[1]))
    localisation_code = localisation_code.strip()

    for year in tqdm.tqdm(range(from_year, until_year + 1)):
        get_synop_data(localisation_code, str(year), os.path.join(synop_data_dir, str(year), 'download'))
        extract_zip_files(os.path.join(synop_data_dir, str(year), 'download'), os.path.join(synop_data_dir, str(year)))
        processed_data = read_synop_data(consts.SYNOP_FEATURES, os.path.join(synop_data_dir, str(year)))
        station_data = pd.concat([station_data, processed_data])
    station_data.to_csv(os.path.join(synop_data_dir, f"{localisation_name}_{localisation_code}_data.csv"), index=False)


def process_auto_station_data(from_year: int, until_year: int, localisation_code: str, localisation_name: str, working_dir: str):
    auto_station_dir = os.path.join(working_dir, 'auto_station_data')
    station_data = pd.DataFrame(columns=AUTO_STATION_CSV_COLUMNS)
    localisation_code = localisation_code.strip()

    for year in tqdm.tqdm(range(from_year, until_year + 1)):
        for month in range(1, 13):
            str_month = prep_zeros_if_needed(str(month), 1)
            if not os.path.exists(os.path.join(auto_station_dir, str(year), str_month)) or len(
                    os.listdir(os.path.join(auto_station_dir, str(year), str_month))) == 0:
                get_auto_station_data(str(year), str_month, os.path.join(auto_station_dir, str(year), str_month, 'download'))
                extract_zip_files(os.path.join(auto_station_dir, str(year), str_month, 'download'),
                                  os.path.join(auto_station_dir, str(year), str_month))

            processed_data = read_auto_station_data(os.path.join(auto_station_dir, str(year), str_month), localisation_code)
            station_data = pd.concat([station_data, processed_data])

    final_df = pd.DataFrame(columns=['date'])
    final_df.set_index('date', inplace=True)
    for param in consts.AUTO_STATION_FEATURES:
        param_rows = station_data.loc[station_data['param_code'] == param[0]]
        if param_rows['value'].isnull().all():
            continue
        values_series = param_rows[['date', 'value']]
        values_series['value'] = [x.replace(',', '.') for x in values_series['value']]
        values_series.rename(columns={'value': param[1]}, inplace=True)
        values_series[param[1]] = values_series[param[1]].astype(float)
        final_df = final_df.join(values_series.set_index('date'), how='outer')

    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'date'}, inplace=True)
    final_df['date'] = pd.to_datetime(final_df['date'])

    final_df.to_csv(os.path.join(auto_station_dir, f"{localisation_name}_{localisation_code}_data.csv"), index=False)


def merge_synop_data_with_auto_station_data(localisation_name: str, localisation_code: str, working_dir: str):
    synop_data_dir = os.path.join(working_dir, 'synop_data')
    auto_station_data_dir = os.path.join(working_dir, 'auto_station_data')
    synop_df = pd.read_csv(os.path.join(synop_data_dir, f"{localisation_name}_{localisation_code}_data.csv"))
    auto_station_df = pd.read_csv(os.path.join(auto_station_data_dir, f"{localisation_name}_{localisation_code}_data.csv"))
    auto_station_df['date'] = pd.to_datetime(auto_station_df['date'])
    synop_df['date'] = pd.to_datetime(synop_df[['year', 'month', 'day', 'hour']])

    add_hourly_wind_velocity(auto_station_df, synop_df, consts.AUTO_WIND[1], consts.VELOCITY_COLUMN[1])
    add_hourly_wind_velocity(auto_station_df, synop_df, consts.AUTO_GUST[1], consts.GUST_COLUMN[1])
    add_hourly_direction(auto_station_df, synop_df)
    add_hourly_precipitation(auto_station_df, synop_df)
    synop_df.to_csv(os.path.join(synop_data_dir, f"{localisation_name}_{localisation_code}_data.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--working_dir', help='Working directory', default=Path(__file__).parent)
    parser.add_argument('--station_code',
                        help='Localisation code for which to get data from stations.'
                             'Codes can be found at https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/wykaz_stacji.csv',
                        required=True, type=str)
    parser.add_argument('--localisation_name',
                        help='Localisation name for which to get data. Will be used to generate final files name.',
                        default=None,
                        type=str)
    parser.add_argument('--start_year', help='Start date for fetching data', type=int, default=2001)
    parser.add_argument('--end_year', help='End date for fetching data', type=int, default=2021)

    args = parser.parse_args()
    station_code = args.station_code
    localisation_name = args.localisation_name

    download_list_of_station(args.working_dir)

    if localisation_name is None:
        localisation_name = resolve_synop_localisation_name(station_code, args.working_dir)

    process_synop_data(args.start_year, args.end_year, station_code, localisation_name, args.working_dir)

    process_auto_station_data(args.start_year, args.end_year, station_code, localisation_name, args.working_dir)

    merge_synop_data_with_auto_station_data(localisation_name, args.station_code, args.working_dir)


if __name__ == "__main__":
    main()

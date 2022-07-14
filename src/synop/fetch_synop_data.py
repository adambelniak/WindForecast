import glob
import math

import numpy as np
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
from util.util import prep_zeros_if_needed

"""This code obtains SYNOP data from 'https://danepubliczne.imgw.pl/'.

    SYNOP file containing meteorological synoptic data for single localisation and year. 

    Wind data is obtained from https://danepubliczne.imgw.pl/datastore/.
    Wind and gusts velocity is obtained by taking an average from 7 observations from HH:30 to HH+1:30 
    Each file from https://danepubliczne.imgw.pl/datastore/ contains data for one month, one parameter for multiple localisations.

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


def download_list_of_station(dir=None):
    file_name = 'wykaz_stacji.csv'

    if dir is None:
        out_file = os.path.join(Path(__file__).parent, 'wykaz_stacji.csv')
    else:
        out_file = os.path.join(dir, 'wykaz_stacji.csv')

    if not os.path.isfile(out_file):
        url = 'https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/' + file_name
        file = requests.get(url, stream=True)
        opened_file = open(out_file, 'wb')
        opened_file.write(file.content)
        opened_file.close()


def resolve_synop_localisation_name(station_code: str, dir=None):
    loc_data = pd.read_csv(os.path.join(dir if dir is not None else Path(__file__).parent, 'wykaz_stacji.csv'),
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

    url = f"https://danepubliczne.imgw.pl/datastore/getfiledown/Arch/Telemetria/Meteo/{year}/Meteo_{year}-{month}.zip"
    if month in ['01', '02'] and year == '2021':
        url = url.replace('zip', 'ZIP')  # ¯\_(ツ)_/¯

    req = requests.get(url, stream=True)
    if req.status_code == 200:
        print(f"Downloading zip to {output_file}")
        with open(output_file, 'wb') as outfile:
            chunk_size = 1048576
            for chunk in req.iter_content(chunk_size=chunk_size):
                outfile.write(chunk)


def extract_zip_files(zip_dir: str, target_dir):
    for file in os.listdir(zip_dir):
        with ZipFile(os.path.join(zip_dir, file), 'r') as zip:
            zip.extractall(path=target_dir)
        os.remove(os.path.join(zip_dir, file))


def read_synop_data(columns, dir='synop_data'):
    station_data = pd.DataFrame(columns=list(list(zip(*columns))[1]))

    for filepath in glob.iglob(rf'{dir}/*.csv', recursive=True):
        synop_data = pd.read_csv(filepath, encoding="ISO-8859-1", header=None)
        required_data = synop_data[list(list(zip(*columns))[0])]
        station_data[list(list(zip(*columns))[1])] = required_data
    return station_data


def read_auto_station_data(dir: str, station_code: str):
    data = pd.DataFrame(columns=AUTO_STATION_CSV_COLUMNS)
    for filepath in glob.iglob(rf'{dir}/*.csv', recursive=True):
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


def process_synop_data(from_year: int, until_year: int, localisation_code: str, localisation_name: str,
                       input_dir=os.path.join(Path(__file__).parent, 'synop_data'),
                       output_dir=os.path.join(Path(__file__).parent, 'synop_data'), columns=None):
    columns = consts.SYNOP_FEATURES if columns is None else columns
    station_data = pd.DataFrame(columns=list(list(zip(*columns))[1]))
    localisation_code = localisation_code.strip()

    for year in tqdm.tqdm(range(from_year, until_year + 1)):
        get_synop_data(localisation_code, str(year), os.path.join(input_dir, str(year), 'download'))
        extract_zip_files(os.path.join(input_dir, str(year), 'download'), os.path.join(input_dir, str(year)))
        processed_data = read_synop_data(columns, os.path.join(input_dir, str(year)))
        station_data = pd.concat([station_data, processed_data])
    station_data.to_csv(os.path.join(output_dir, f"{localisation_name}_{localisation_code}_data.csv"), index=False)


def process_auto_station_data(from_year: int, until_year: int, localisation_code: str, localisation_name: str,
                              input_dir=os.path.join(Path(__file__).parent, 'auto_station_data'),
                              output_dir=os.path.join(Path(__file__).parent, 'auto_station_data')):
    station_data = pd.DataFrame(columns=AUTO_STATION_CSV_COLUMNS)
    localisation_code = localisation_code.strip()

    for year in tqdm.tqdm(range(from_year, until_year + 1)):
        for month in range(1, 13):
            str_month = prep_zeros_if_needed(str(month), 1)
            if not os.path.exists(os.path.join(input_dir, str(year), str_month)) or len(
                    os.listdir(os.path.join(input_dir, str(year), str_month))) == 0:
                get_auto_station_data(str(year), str_month, os.path.join(input_dir, str(year), str_month, 'download'))
                extract_zip_files(os.path.join(input_dir, str(year), str_month, 'download'),
                                  os.path.join(input_dir, str(year), str_month))

            processed_data = read_auto_station_data(os.path.join(input_dir, str(year), str_month), localisation_code)
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

    final_df.to_csv(os.path.join(output_dir, f"{localisation_name}_{localisation_code}_data.csv"), index=False)


def merge_synop_data_with_auto_station_data(localisation_name: str, localisation_code: str,
                                            synop_data_dir = os.path.join(Path(__file__).parent, 'synop_data'),
                                            auto_station_data_dir = os.path.join(os.path.join(Path(__file__).parent, 'auto_station_data'))):
    synop_df = pd.read_csv(os.path.join(synop_data_dir, f"{localisation_name}_{localisation_code}_data.csv"))
    auto_station_df = pd.read_csv(os.path.join(auto_station_data_dir, f"{localisation_name}_{localisation_code}_data.csv"))

    add_hourly_wind_velocity(synop_df, auto_station_df, consts.AUTO_WIND[1], consts.VELOCITY_COLUMN[1])
    add_hourly_wind_velocity(synop_df, auto_station_df, consts.AUTO_GUST[1], consts.GUST_COLUMN[1])
    add_hourly_direction(synop_df, auto_station_df)
    synop_df.to_csv(os.path.join(synop_data_dir, f"{localisation_name}_{localisation_code}_data.csv"), index=False)


def add_hourly_wind_velocity(synop_df: pd.DataFrame, auto_station_df: pd.DataFrame, auto_station_param: str, synop_param: str):
    copy_auto_station_df = pd.DataFrame(auto_station_df)
    copy_synop_date = pd.DataFrame(synop_df)
    if auto_station_param == consts.AUTO_GUST[1]:
        copy_auto_station_df[auto_station_param] = auto_station_df[auto_station_param].rolling(6, min_periods=1).max()
    else:
        auto_station_wind_data = np.array(auto_station_df[auto_station_param].tolist())
        convolved = np.convolve(auto_station_wind_data, np.full(6, 1/6), 'valid')
        convolved = np.insert(convolved, 0, auto_station_wind_data[0:3])
        convolved = np.append(convolved, auto_station_wind_data[-2:])
        copy_auto_station_df['convolved'] = convolved

    copy_auto_station_df['date'] = pd.to_datetime(copy_auto_station_df['date'])
    copy_synop_date['date'] = pd.to_datetime(synop_df[['year', 'month', 'day', 'hour']])

    values = []
    for date in copy_synop_date['date'].tolist():
        auto_station_row = copy_auto_station_df.loc[copy_auto_station_df['date'] == date]
        if len(auto_station_row) == 0:
            print(f"Auto station data not found for date {date}.")
            values.append(copy_synop_date.loc[copy_synop_date['date'] == date][synop_param].item())
        else:
            if auto_station_param == consts.AUTO_GUST[1]:
                values.append("{:.1f}".format(auto_station_row[auto_station_param].item()))
            else:
                values.append("{:.1f}".format(auto_station_row['convolved'].item()))

    synop_df[synop_param] = values


def add_hourly_direction(synop_df, auto_station_df):
    copy_auto_station_df = pd.DataFrame(auto_station_df)
    copy_synop_date = pd.DataFrame(synop_df)
    auto_station_wind_data = np.array(auto_station_df[consts.AUTO_WIND_DIRECTION[1]].tolist())
    cosinus = np.vectorize(lambda x: math.cos(math.pi * (x/180)))(auto_station_wind_data)
    sinus = np.vectorize(lambda x: math.sin(math.pi * (x/180)))(auto_station_wind_data)

    convolved_cosinus = np.convolve(cosinus, np.full(6, 1 / 6), 'valid')
    convolved_cosinus = np.insert(convolved_cosinus, 0, cosinus[0:3])
    convolved_cosinus = np.append(convolved_cosinus, cosinus[-2:])

    convolved_sinus = np.convolve(sinus, np.full(6, 1 / 6), 'valid')
    convolved_sinus = np.insert(convolved_sinus, 0, sinus[0:3])
    convolved_sinus = np.append(convolved_sinus, sinus[-2:])

    copy_auto_station_df['convolved_cosinus'] = convolved_cosinus
    copy_auto_station_df['convolved_sinus'] = convolved_sinus
    copy_auto_station_df['date'] = pd.to_datetime(copy_auto_station_df['date'])
    copy_synop_date['date'] = pd.to_datetime(synop_df[['year', 'month', 'day', 'hour']])

    values = []
    for date in copy_synop_date['date'].tolist():
        auto_station_row = copy_auto_station_df.loc[copy_auto_station_df['date'] == date]
        if len(auto_station_row) == 0:
            print(f"Auto station data not found for date {date}.")
            values.append(copy_synop_date.loc[copy_synop_date['date'] == date][consts.DIRECTION_COLUMN[1]].item())
        else:
            convolved_cos = auto_station_row['convolved_cosinus']
            convolved_sin = auto_station_row['convolved_sinus']

            angle = math.atan2(convolved_sin.item(), convolved_cos.item())
            angle *= 180 / math.pi
            if angle < 0: angle += 360
            values.append("{:.1f}".format(angle))

    synop_df[consts.DIRECTION_COLUMN[1]] = values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', help='Working directory', default=None)
    parser.add_argument('--out', help='Directory where to save synop files', default='synop_data')
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

    parser.add_argument('--plot_box_year', help='Year fow which create box plot for each month', type=int, default=2019)

    args = parser.parse_args()
    station_code = args.station_code
    localisation_name = args.localisation_name

    download_list_of_station(args.dir)

    if localisation_name is None:
        localisation_name = resolve_synop_localisation_name(station_code, args.dir)

    process_synop_data(args.start_year, args.end_year, station_code, localisation_name)

    process_auto_station_data(args.start_year, args.end_year, station_code, localisation_name)

    merge_synop_data_with_auto_station_data(localisation_name, args.station_code)


if __name__ == "__main__":
    main()
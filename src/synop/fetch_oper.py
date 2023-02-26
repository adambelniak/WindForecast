import argparse
import os
import re
from datetime import datetime
import pandas as pd
from pathlib import Path
import requests
from bs4 import BeautifulSoup

from synop.consts import TEMPERATURE, DEW_POINT, HUMIDITY, DIRECTION_COLUMN, VELOCITY_COLUMN, GUST_COLUMN, PRESSURE, \
    PRESSURE_AT_SEA_LEVEL, PRECIPITATION_6H, CLOUD_COVER, LOWER_CLOUDS, VISIBILITY, AUTO_HOUR_PRECIPITATION, \
    AUTO_WIND_DIRECTION, AUTO_WIND, AUTO_GUST
from synop.tele_util import add_hourly_wind_velocity, add_hourly_direction, add_hourly_precipitation

OGIMET_URL_TEMPLATE = "https://ogimet.com/cgi-bin/gsynres?ind={0}&lang=en&decoded=yes&ndays=1&ano={1}&mes={2}&day={3}&hora={4}"
TELEMETRY_URL = "https://hydro.imgw.pl/api/station/meteo/?id={0}"

TELEMETRY_MAPPING = {
    AUTO_HOUR_PRECIPITATION[1]: 'hourlyPrecipRecords',
    AUTO_WIND_DIRECTION[1]: 'windDirectionTelRecords',
    AUTO_WIND[1]: 'windVelocityTelRecords',
    AUTO_GUST[1]: 'windMaxVelocityRecords'
}

ogimet_table_structure = [
    "date",
    "time",
    TEMPERATURE[1],
    DEW_POINT[1],
    HUMIDITY[1],
    "tmax",
    "tmin",
    DIRECTION_COLUMN[1],
    VELOCITY_COLUMN[1],
    GUST_COLUMN[1],
    PRESSURE[1],
    PRESSURE_AT_SEA_LEVEL[1],
    "pressure_tendency",
    PRECIPITATION_6H[1],
    CLOUD_COVER[1],
    LOWER_CLOUDS[1],
    "cloud_base",
    VISIBILITY[1]
]

# This data is not needed or will be taken from telemetric measures
columns_to_be_dropped = ['time', 'tmax', 'tmin', DIRECTION_COLUMN[1], VELOCITY_COLUMN[1], GUST_COLUMN[1], PRESSURE_AT_SEA_LEVEL[1],
                         "pressure_tendency", PRECIPITATION_6H[1], "cloud_base"]


def fill_df_with_telemetric_data(df, tele_station_code: str) -> pd.DataFrame:
    url = TELEMETRY_URL.format(tele_station_code)

    response = requests.get(url)
    if not response.ok:
        return None

    content = response.json()
    data = content[TELEMETRY_MAPPING[AUTO_WIND[1]]]
    wind_velocity_df = pd.DataFrame({'date': [datetime.strptime(row['date'], '%Y-%m-%dT%H:%M:%SZ') for row in data],
                       AUTO_WIND[1]: [row['value'] for row in data]})

    df = add_hourly_wind_velocity(wind_velocity_df, df, AUTO_WIND[1], VELOCITY_COLUMN[1])

    data = content[TELEMETRY_MAPPING[AUTO_GUST[1]]]
    wind_gust_df = pd.DataFrame({'date': [datetime.strptime(row['date'], '%Y-%m-%dT%H:%M:%SZ') for row in data],
                                     AUTO_GUST[1]: [row['value'] for row in data]})
    df = add_hourly_wind_velocity(wind_gust_df, df, AUTO_GUST[1], GUST_COLUMN[1])

    data = content[TELEMETRY_MAPPING[AUTO_WIND_DIRECTION[1]]]
    wind_direction_df = pd.DataFrame({'date': [datetime.strptime(row['date'], '%Y-%m-%dT%H:%M:%SZ') for row in data],
                                 AUTO_WIND_DIRECTION[1]: [row['value'] for row in data]})
    df = add_hourly_direction(wind_direction_df, df)

    data = content[TELEMETRY_MAPPING[AUTO_HOUR_PRECIPITATION[1]]]
    precipitation_df = pd.DataFrame({'date': [datetime.strptime(row['date'], '%Y-%m-%dT%H:%M:%SZ') for row in data],
                                      AUTO_HOUR_PRECIPITATION[1]: [row['value'] for row in data]})
    df = add_hourly_precipitation(precipitation_df, df)

    return df


def fetch_recent_synop(station_code: str, tele_station_code: str) -> pd.DataFrame:
    with open(os.path.join(Path(__file__).parent, "..", "gfs_oper", "processed", "available_starting_points.txt"), 'r') as f:
        for line in f:
            pass
        latest_available_starting_point = line

    final_csv = os.path.join(Path(__file__).parent, "oper_data", latest_available_starting_point, "data.csv")

    if os.path.exists(final_csv):
        return pd.read_csv(final_csv)

    date_matcher = re.match(r'(\d{4})(\d{2})(\d{2})(\d{2})', latest_available_starting_point)
    ogimet_url = OGIMET_URL_TEMPLATE.format(station_code, date_matcher.group(1), date_matcher.group(2),
                                            date_matcher.group(3), date_matcher.group(4))

    response = requests.get(ogimet_url)
    if not response.ok:
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    data_table = soup.find_all('table')[2]
    table_rows = data_table.thead.find_all('tr')[1:]
    rows_key = ogimet_table_structure
    if len(table_rows[0].find_all('td')) == 23:  # snow and inso included
        rows_key.insert(17, "inso")
        rows_key.insert(19, "snow")

    data = {}
    for row in table_rows:
        columns = row.find_all('td')
        for index, key in enumerate(rows_key):
            if key not in data.keys():
                data[key] = []
            data[key].append(columns[index].text)

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.drop(columns=columns_to_be_dropped, inplace=True)

    fill_df_with_telemetric_data(df, tele_station_code)
    os.makedirs(os.path.dirname(final_csv), exist_ok=True)
    df.to_csv(final_csv)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--station_code',
                        help='Localisation code for which to get data from stations.'
                             'Codes can be found at https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/wykaz_stacji.csv',
                        required=True, type=str)
    parser.add_argument('--tele_station_code',
                        help='Localisation code for telemetric station which to get data from.',
                        required=True, type=str)
    parser.add_argument('--localisation_name',
                        help='Localisation name for which to get data. Will be used to generate final files name.',
                        default=None,
                        type=str)
    args = parser.parse_args()
    fetch_recent_synop(args.station_code, args.tele_station_code)


if __name__ == "__main__":
    main()

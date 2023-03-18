import numpy as np
import math
import pandas as pd

from synop import consts
from synop.consts import AUTO_HOUR_PRECIPITATION, PRECIPITATION, PRECIPITATION_6H


def add_hourly_wind_velocity(auto_station_df: pd.DataFrame, synop_df: pd.DataFrame, auto_station_param: str, synop_param: str):
    copy_auto_station_df = pd.DataFrame(auto_station_df)
    copy_synop_df = pd.DataFrame(synop_df)
    if auto_station_param == consts.AUTO_GUST[1]:
        copy_auto_station_df[auto_station_param] = auto_station_df[auto_station_param].rolling(6, min_periods=1).max()
    else:
        auto_station_wind_data = np.array(auto_station_df[auto_station_param].tolist())
        convolved = np.convolve(auto_station_wind_data, np.full(6, 1/6), 'valid')
        convolved = np.insert(convolved, 0, auto_station_wind_data[0:3])
        convolved = np.append(convolved, auto_station_wind_data[-2:])
        copy_auto_station_df['convolved'] = convolved

    values = []
    for date in copy_synop_df['date'].tolist():
        auto_station_row = copy_auto_station_df.loc[copy_auto_station_df['date'] == date]
        if len(auto_station_row) == 0:
            print(f"Auto station data not found for date {date}.")
            values.append(copy_synop_df.loc[copy_synop_df['date'] == date][synop_param].item())
        else:
            if auto_station_param == consts.AUTO_GUST[1]:
                values.append("{:.1f}".format(auto_station_row[auto_station_param].item()))
            else:
                values.append("{:.1f}".format(auto_station_row['convolved'].item()))

    synop_df[synop_param] = values

    return synop_df


def add_hourly_direction(auto_station_df: pd.DataFrame, synop_df: pd.DataFrame):
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

    return synop_df


def add_hourly_precipitation(auto_station_df: pd.DataFrame, synop_df: pd.DataFrame) -> pd.DataFrame:
    copy_auto_station_df = pd.DataFrame(auto_station_df)
    copy_synop_date = pd.DataFrame(synop_df)

    values = []
    for date in copy_synop_date['date'].tolist():
        auto_station_row = copy_auto_station_df.loc[copy_auto_station_df['date'] == date]
        if len(auto_station_row) == 0:
            print(f"Auto station data not found for date {date}.")
            if PRECIPITATION[1] in copy_synop_date.columns:
                values.append(copy_synop_date.loc[copy_synop_date['date'] == date][PRECIPITATION[1]].item())
            else:
                values.append(0)
        else:
            values.append("{:.1f}".format(auto_station_row[AUTO_HOUR_PRECIPITATION[1]].item()))
    # TODO it should be precipitation
    synop_df[PRECIPITATION_6H[1]] = values

    return synop_df

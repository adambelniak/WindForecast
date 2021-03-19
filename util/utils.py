from datetime import datetime

import numpy as np
import pytz


def convert_wind(single_gfs, u_wind_label, v_wind_label):
    single_gfs["velocity"] = np.sqrt(single_gfs[u_wind_label] ** 2 + single_gfs[v_wind_label] ** 2)
    single_gfs = single_gfs.drop([u_wind_label, v_wind_label], axis=1)

    return single_gfs


def utc_to_local(date):
    return date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Warsaw')).replace(tzinfo=None)

# if __name__ == "__main__":
#     print(datetime(2015, 4, 1, 12, 20).replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Warsaw')).replace(tzinfo=None))
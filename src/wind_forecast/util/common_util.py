from enum import Enum

import pytz
import numpy as np


def prep_zeros_if_needed(value: str, number_of_zeros: int):
    for i in range(number_of_zeros - len(value) + 1):
        value = '0' + value
    return value


def utc_to_local(date):
    return date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Warsaw')).replace(tzinfo=None)


def declination_of_earth(date):
    day_of_year = date.timetuple().tm_yday
    return 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + day_of_year) / 365.0))


class NormalizationType(Enum):
    STANDARD = 0
    MINMAX = 1

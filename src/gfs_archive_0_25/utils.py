import math

from util.coords import Coords


def get_nearest_coords(coords: Coords) -> Coords:
    lat1 = math.floor(coords.nlat * 4) / 4
    lat2 = math.floor((coords.nlat + 0.25) * 4) / 4
    if lat2 > 90:
        lat2 = 90
    long1 = math.floor(coords.elon * 4) / 4
    long2 = math.floor((coords.elon + 0.25) * 4) / 4
    if long2 >= 360:
        long2 = 0
    return Coords(lat2, lat1, long1, long2)


def prep_zeros_if_needed(value: str, number_of_zeros: int):
    for i in range(number_of_zeros - len(value) + 1):
        value = '0' + value
    return value

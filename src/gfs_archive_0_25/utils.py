import math


def get_nearest_coords(latitude, longitude):
    lat1 = math.floor(latitude * 4) / 4
    lat2 = math.floor((latitude + 0.25) * 4) / 4
    if lat2 > 90:
        lat2 = 90
    long1 = math.floor(longitude * 4) / 4
    long2 = math.floor((longitude + 0.25) * 4) / 4
    if long2 >= 360:
        long2 = 0
    return [[lat1, lat2], [long1, long2]]


def prep_zeros_if_needed(value, number_of_zeros):
    for i in range(number_of_zeros - len(value) + 1):
        value = '0' + value
    return value

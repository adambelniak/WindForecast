import argparse
import sys
from datetime import datetime
import pandas as pd

sys.path.insert(0, '../rda-apps-clients/src/python')
sys.path.insert(1, '..')

from geopy.geocoders import Nominatim, GeoNames
import rdams_client as rc
from tqdm import tqdm
import os
from utils import get_nearest_coords
from enum import Enum

req_id_path = 'req_list.csv'

class RequestStatus(Enum):
    SENT = 'sent'
    FAILED = 'failed'
    COMPLETED = 'completed'


def save_request_id(req_id: str):
    if not os.path.isfile(req_id_path):
        pseudo_db = pd.DataFrame(columns=["req_id", "status"])
    else:
        pseudo_db = pd.read_csv(req_id_path)
    pseudo_db.append({"req_id": req_id, "status": RequestStatus.SENT}, ignore_index=True)
    pseudo_db.to_csv(req_id_path)


def generate_product_description(start_hour, end_hour, step=3):
    product = ''
    for x in range(start_hour, end_hour + step, step):
        product = product + '{}-hour Forecast/'.format(x)
    return product[:-1]


def find_coordinates(path, output_file_name="city_geo.csv"):
    location_list = pd.read_csv(path, encoding="ISO-8859-1",
                                names=['id', 'city_name', 'meteo_code'])

    city_list = location_list["city_name"].to_list()
    geolocator = Nominatim(user_agent='gfs_fetch_processor')
    geo_list = []

    print("Downloading coordinates for provided cities")
    for city_name in tqdm(city_list):
        geo = geolocator.geocode(city_name)
        if geo:
            geo_list.append([city_name, geo.latitude, geo.longitude])

    path_to_write = os.path.join("../city_coordinates", output_file_name)
    data = pd.DataFrame(geo_list, columns=["city_name", "latitude", "longitude"])
    data.to_csv(path_to_write)
    return data


def build_template(latitude, longitude, start_date, end_date, param_code, product, level='HTGL:10'):
    end_date = end_date.strftime('%Y%m%d%H%H%M%M')
    start_date = start_date.strftime('%Y%m%d%H%H%M%M')
    date = '{}/to/{}'.format(start_date, end_date)
    control = {
        'dataset': 'ds084.1',
        'date': date,
        'param': param_code,
        'level': level,
        'oformat': 'csv',
        'nlat': latitude,
        'slat': latitude,
        'elon': longitude,
        'wlon': longitude,
        'product': product
    }

    return control


def transform_coordinates(data):
    """
    Find nearest GFS coordinates for provided cities. Filter duplicates
    :param data: Pandas dataFrame
    :return:
    """
    coordinates = data.apply(lambda x: get_nearest_coords(x["latitude"], x["longitude"]), axis=1)
    coordinates = [[x[0], y[0]] for x, y in coordinates]
    data[["latitude", "longitude"]] = coordinates
    before_duplicates_filter = len(data)
    gfs_coordinates = data.drop_duplicates(subset=["latitude", "longitude"])
    num_dup = before_duplicates_filter - len(gfs_coordinates)

    print("Removed {} duplicates rows".format(num_dup))

    return data, gfs_coordinates[["latitude", "longitude"]].values


def run_gfs_processor(**kwargs):
    if kwargs["fetch_city_coordinates"]:
        data = find_coordinates(kwargs["citi_list"])
    else:
        data = pd.read_csv(kwargs["coordinate_path"])
    data, gfs_coordinates = transform_coordinates(data)
    start_date = datetime.strptime(kwargs["start_date"], '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(kwargs["end_date"], '%Y-%m-%d %H:%M')
    product = generate_product_description(kwargs['forecast_start'], kwargs['forecast_end'])

    data.to_csv('map_cities_to_gfs_cords')

    for latitude, longitude in gfs_coordinates:
        template = build_template(latitude, longitude, start_date, end_date, 'V GRD', product)
        response = rc.submit_json(template)
        assert response['status'] == 'ok'
        rqst_id = response['result']['request_id']
        save_request_id(rqst_id)
        print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fetch_city_coordinates', help='get coordinates for provided cities', default=False)
    parser.add_argument('--citi_list', help='Path to SYNOP list with locations names', default=False)
    parser.add_argument('--coordinate_path', help='Path to list of cities with coordinates',
                        default='../city_coordinates/city_geo_test.csv')
    parser.add_argument('--start_date', help='Start date GFS', default='2015-01-15 00:00')
    parser.add_argument('--end_date', help='End date GFS', default='2020-08-01 00:00')
    parser.add_argument('--gfs_parameter', help='Parameter to process from NCAR', type=str)
    parser.add_argument('--forecast_start', default=3)
    parser.add_argument('--forecast_end', default=168)

    args = parser.parse_args()
    run_gfs_processor(**vars(args))

import argparse
import sys
from datetime import datetime
import pandas as pd
import schedule

sys.path.insert(0, '../rda-apps-clients/src/python')
sys.path.insert(1, '..')

from geopy.geocoders import Nominatim, GeoNames
from tqdm import tqdm
import os
from utils import get_nearest_coords
from enum import Enum
from own_logger import logger
from rdams_client import submit_json
from typing import Optional

REQ_ID_PATH = 'req_list.csv'


class RequestStatus(Enum):
    PENDING = 'pending'
    SENT = 'sent'
    FAILED = 'failed'
    COMPLETED = 'completed'


def save_request(latitude: str, longitude: str, status: RequestStatus, req_id: Optional[str] = None):
    if not os.path.isfile(REQ_ID_PATH):
        pseudo_db = pd.DataFrame(columns=["req_id", "status"])
    else:
        pseudo_db = pd.read_csv(REQ_ID_PATH, index_col=[0])
    pseudo_db = pseudo_db.append(
        {"req_id": req_id, "status": status.value, "latitude": latitude, "longitude": longitude},
        ignore_index=True)
    pseudo_db.to_csv(REQ_ID_PATH)


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

    logger.info("Downloading coordinates for provided cities")
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


def prepare_coordinates(data):
    """
    Round GFS coordinates for provided cities. Filter duplicates.
    :param data: Pandas dataFrame
    :return:
    """
    coordinates = data.apply(lambda x: [round(x["latitude"], 1), round(x["longitude"], 1)], axis=1)
    data[["latitude", "longitude"]] = [x for x in coordinates]
    before_duplicates_filter = len(data)
    data = data.drop_duplicates(subset=["latitude", "longitude"])
    num_dup = before_duplicates_filter - len(data)

    logger.info("Removed {} duplicates rows".format(num_dup))

    return data


def prepare_data(**kwargs):
    if kwargs["fetch_city_coordinates"]:
        data = find_coordinates(kwargs["city_list"])
    else:
        data = pd.read_csv(kwargs["coordinate_path"])
    data = prepare_coordinates(data)
    start_date = datetime.strptime(kwargs["start_date"], '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(kwargs["end_date"], '%Y-%m-%d %H:%M')
    product = generate_product_description(kwargs['forecast_start'], kwargs['forecast_end'])

    data.to_csv('map_cities_to_gfs_cords.csv')

    for latitude, longitude in data[['latitude', 'longitude']].values:
        save_request(latitude, longitude, RequestStatus.PENDING)
    return data[['latitude', 'longitude']].values


def gfs_request_sender(kwargs):

    start_date = datetime.strptime(kwargs["start_date"], '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(kwargs["end_date"], '%Y-%m-%d %H:%M')
    product = generate_product_description(kwargs['forecast_start'], kwargs['forecast_end'])
    param = kwargs['gfs_parameter']
    request_db = pd.read_csv(REQ_ID_PATH, index_col=0)

    request_to_sent = request_db[request_db["status"] == RequestStatus.PENDING.value]

    for index, request in request_to_sent.iterrows():
        latitude = request["latitude"]
        longitude = request["longitude"]

        template = build_template(latitude, longitude, start_date, end_date, param, product)
        response = submit_json(template)
        if response['status'] == 'ok':
            reqst_id = response['result']['request_id']
            request_db.loc[index]["status"] = RequestStatus.SENT.value
            request_db.loc[index]["req_id"] = reqst_id
        else:
            logger.info("Rda has returned error")
        logger.info(response)
    request_db.to_csv(REQ_ID_PATH)


def prepare_and_start_processor(**kwargs):
    gfs_coordinates = prepare_data(**kwargs)
    gfs_request_sender(kwargs)
    # try:
    #     schedule.every(30).seconds.do(gfs_request_sender, kwargs)
    # except Exception as e:
    #     logger.error(e, exc_info=True)

    # while True:
    #     schedule.run_pending()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fetch_city_coordinates', help='get coordinates for provided cities', default=False)
    parser.add_argument('--city_list', help='Path to SYNOP list with locations names', default=False)
    parser.add_argument('--coordinate_path', help='Path to list of cities with coordinates',
                        default='../city_coordinates/city_geo_test.csv')
    parser.add_argument('--start_date', help='Start date GFS', default='2015-01-15 00:00')
    parser.add_argument('--end_date', help='End date GFS', default='2015-01-21 00:00')
    parser.add_argument('--gfs_parameter', help='Parameter to process from NCAR', type=str, default='V GRD')
    parser.add_argument('--forecast_start', default=3)
    parser.add_argument('--forecast_end', default=168)

    args = parser.parse_args()
    prepare_and_start_processor(**vars(args))

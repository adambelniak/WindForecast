import argparse
import json
from datetime import datetime
import time
from pathlib import Path

import pandas as pd
import schedule
from geopy.geocoders import Nominatim
from tqdm import tqdm
from enum import Enum
from gfs_archive_0_25.gfs_processor.own_logger import logger
from gfs_archive_0_25.gfs_processor.rdams_client import submit_json
from gfs_archive_0_25.gfs_processor.consts import *

REQ_ID_PATH = os.path.join(Path(__file__).parent, 'csv/req_list.csv')


class RequestStatus(Enum):
    PENDING = 'Pending'
    SENT = 'Sent'
    FAILED = 'Failed'  # HTTP request has failed
    ERROR = 'Error'  # HTTP request succeeded, but request processing failed
    COMPLETED = 'Completed'
    DOWNLOADED = 'Downloaded'
    FINISHED = 'Finished'
    PURGED = 'Purged'


class RequestType(Enum):
    BULK = 'Bulk'
    POINT = 'Point'


def point_coords_to_region_coords(coords):
    region_coords = pd.DataFrame(columns=[NLAT_FIELD, SLAT_FIELD, WLON_FIELD, ELON_FIELD])
    region_coords[[NLAT_FIELD, SLAT_FIELD, WLON_FIELD, ELON_FIELD]] = [
        [x[1]['latitude'], x[1]['latitude'], x[1]['longitude'], x[1]['longitude']] for x in coords.iterrows()]
    return region_coords


def save_request_to_pseudo_db(request_type: RequestType, request_status: RequestStatus, **kwargs):
    if not os.path.isfile(REQ_ID_PATH):
        pseudo_db = pd.DataFrame(
            columns=[REQUEST_ID_FIELD, REQUEST_TYPE_FIELD, REQUEST_STATUS_FIELD, NLAT_FIELD, SLAT_FIELD, WLON_FIELD,
                     ELON_FIELD, PARAM_FIELD, LEVEL_FIELD, HOURS_TYPE_FIELD])
    else:
        pseudo_db = pd.read_csv(REQ_ID_PATH, index_col=[0])
    pseudo_db = pseudo_db.append(
        {REQUEST_ID_FIELD: kwargs[REQUEST_ID_FIELD],
         REQUEST_TYPE_FIELD: request_type.value,
         REQUEST_STATUS_FIELD: request_status.value,
         NLAT_FIELD: kwargs[NLAT_FIELD],
         SLAT_FIELD: kwargs[SLAT_FIELD],
         WLON_FIELD: kwargs[WLON_FIELD],
         ELON_FIELD: kwargs[ELON_FIELD],
         PARAM_FIELD: kwargs[PARAM_FIELD],
         LEVEL_FIELD: kwargs[LEVEL_FIELD],
         HOURS_TYPE_FIELD: kwargs[HOURS_TYPE_FIELD]
         }, ignore_index=True)
    logger.info(
        f"Saving a new request of type {request_type} for coords (lat: {kwargs[NLAT_FIELD]}-{kwargs[SLAT_FIELD]}, "
        f"lon: {kwargs[WLON_FIELD]}-{kwargs[ELON_FIELD]}), param {kwargs[PARAM_FIELD]}, level {kwargs[LEVEL_FIELD]}, hours_type {kwargs[HOURS_TYPE_FIELD]}...")
    pseudo_db.to_csv(REQ_ID_PATH)


def generate_product_description(start_hour, end_hour, hours_type, step=3):
    product = ''
    for x in range(start_hour, end_hour + step, step):
        if hours_type in ['average', 'all']:
            if x % 2 == 0:
                product = product + '3-hour Average (initial+{0} to initial+{1})/'.format(x, x + step)
                product = product + '6-hour Average (initial+{0} to initial+{1})/'.format(x, x + 2 * step)

        if hours_type in ['point', 'all']:
            product = product + '{}-hour Forecast/'.format(x)
    return product[:-1]


def find_coordinates(path, output_file_name="city_geo.csv"):
    location_list = pd.read_csv(path, encoding="ISO-8859-1",
                                names=['city_name', 'meteo_code'])

    city_list = location_list["city_name"].to_list()
    geolocator = Nominatim(user_agent='gfs_fetch_processor')
    geo_list = []

    logger.info("Downloading coordinates for provided cities")
    for city_name in tqdm(city_list):
        geo = geolocator.geocode(city_name)
        if geo:
            geo_list.append([city_name, geo.latitude, geo.longitude])

    path_to_write = os.path.join(Path(__file__).parent, "city_coordinates", output_file_name)
    data = pd.DataFrame(geo_list, columns=["city_name", "latitude", "longitude"])
    data.to_csv(path_to_write)
    return data


def build_template(nlat, slat, elon, wlon, start_date, end_date, param_code, product, level, format):
    end_date = end_date.strftime('%Y%m%d%H%H%M%M')
    start_date = start_date.strftime('%Y%m%d%H%H%M%M')
    date = '{}/to/{}'.format(start_date, end_date)
    control = {
        'dataset': 'ds084.1',
        'date': date,
        'param': param_code,
        'oformat': format,
        'nlat': nlat,
        'slat': slat,
        'elon': elon,
        'wlon': wlon,
        'product': product
    }

    if level != 'Def':
        control['level'] = level

    return control


def prepare_coordinates(coords_data):
    """
    Round GFS coordinates for provided data. Filter duplicates.
    :param coords_data: Pandas dataFrame
    :return:
    """
    coordinates = coords_data.apply(
        lambda x: [round(x[NLAT_FIELD], 1), round(x[SLAT_FIELD], 1), round(x[WLON_FIELD], 1), round(x[ELON_FIELD], 1)],
        axis=1)
    coords_data[[NLAT_FIELD, SLAT_FIELD, WLON_FIELD, ELON_FIELD]] = [x for x in coordinates]
    before_duplicates_filter = len(coords_data)
    coords_data = coords_data.drop_duplicates(subset=[NLAT_FIELD, SLAT_FIELD, WLON_FIELD, ELON_FIELD])
    num_dup = before_duplicates_filter - len(coords_data)

    logger.info("Removed {} duplicates rows".format(num_dup))

    return coords_data


def read_params_from_input_file(path):
    logger.info("Reading gfs parameters from " + path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        raise Exception("Input path does not exist.")


def prepare_bulk_region_request(params_to_fetch, **kwargs):
    coords_data = pd.DataFrame(columns=[NLAT_FIELD, SLAT_FIELD, WLON_FIELD, ELON_FIELD])
    coords_data = coords_data.append(
        {
            NLAT_FIELD: kwargs[NLAT_FIELD],
            SLAT_FIELD: kwargs[SLAT_FIELD],
            WLON_FIELD: kwargs[WLON_FIELD],
            ELON_FIELD: kwargs[ELON_FIELD]
        }, ignore_index=True)
    coords_data = prepare_coordinates(coords_data)

    for param in params_to_fetch:
        param, level, hours_type = param[PARAM_FIELD], param[LEVEL_FIELD], param[HOURS_TYPE_FIELD]
        if level == '':
            level = 'Def'
        for nlat, slat, wlon, elon in coords_data[[NLAT_FIELD, SLAT_FIELD, WLON_FIELD, ELON_FIELD]].values:
            save_request_to_pseudo_db(RequestType.BULK, RequestStatus.PENDING, nlat=nlat, slat=slat,
                                      elon=elon, wlon=wlon, param=param, level=level, hours_type=hours_type,
                                      request_id=None)


def prepare_points_request(params_to_fetch, **kwargs):
    if kwargs["fetch_city_coordinates"]:
        coords_data = find_coordinates(kwargs["city_list"])
    else:
        coords_data = pd.read_csv(kwargs["coordinate_path"])

    coords_data = prepare_coordinates(point_coords_to_region_coords(coords_data))

    for param in params_to_fetch:
        param, level, hours_type = param[PARAM_FIELD], param[LEVEL_FIELD], param[HOURS_TYPE_FIELD]
        if level == '':
            level = 'Def'
        for nlat, slat, wlon, elon in coords_data[[NLAT_FIELD, SLAT_FIELD, WLON_FIELD, ELON_FIELD]].values:
            save_request_to_pseudo_db(RequestType.POINT, RequestStatus.PENDING, nlat=nlat, slat=slat,
                                      elon=elon, wlon=wlon, param=param, level=level, hours_type=hours_type,
                                      request_id=None)


def prepare_requests(**kwargs):
    if kwargs['input_file'] is not None:
        params_to_fetch = read_params_from_input_file(kwargs['input_file'])
        logger.info(f"Preparing requests for {len(params_to_fetch)} parameters...")
    else:
        params_to_fetch = [{PARAM_FIELD: kwargs['gfs_parameter'], "level": kwargs['gfs_level'],
                            HOURS_TYPE_FIELD: kwargs[HOURS_TYPE_FIELD]}]

    if kwargs["bulk"]:
        prepare_bulk_region_request(params_to_fetch, **kwargs)
    else:
        prepare_points_request(params_to_fetch, **kwargs)


def send_prepared_requests(kwargs):
    start_date = datetime.strptime(kwargs["start_date"], '%Y-%m-%d %H:%M')
    end_date = datetime.strptime(kwargs["end_date"], '%Y-%m-%d %H:%M')
    request_db = pd.read_csv(REQ_ID_PATH, index_col=0)

    requests_to_send = request_db[request_db[REQUEST_STATUS_FIELD] == RequestStatus.PENDING.value]

    for index, request in requests_to_send.iterrows():
        nlat, slat, elon, wlon = request[[NLAT_FIELD, SLAT_FIELD, ELON_FIELD, WLON_FIELD]]
        request_type = request[REQUEST_TYPE_FIELD]
        param = request[PARAM_FIELD]
        level = request[LEVEL_FIELD]
        hours_type = request[HOURS_TYPE_FIELD]
        product = generate_product_description(kwargs['forecast_start'], kwargs['forecast_end'], hours_type=hours_type)

        template = build_template(nlat, slat, elon, wlon, start_date, end_date, param, product, level,
                                  'csv' if request_type == RequestType.POINT.value else 'netCDF')
        response = submit_json(template)
        if response['status'] == 'ok':
            request_id = response['result']['request_id']
            request_db.loc[index, REQUEST_STATUS_FIELD] = RequestStatus.SENT.value
            request_db.loc[index, REQUEST_ID_FIELD] = str(int(request_id))
        else:
            logger.info("Rda has returned error.")
            if response['status'] == 'error' and TOO_MANY_REQUESTS in response['messages']:
                logger.info("Too many requests. Request will be sent on next scheduler trigger.")
                break
            else:
                request_db.loc[index, REQUEST_STATUS_FIELD] = RequestStatus.FAILED.value
        logger.info(response)
    request_db.to_csv(REQ_ID_PATH)
    logger.info("Sending requests done. Waiting for next scheduler trigger.")


def prepare_and_start_processor(**kwargs):
    if kwargs['send_only'] is False:
        prepare_requests(**kwargs)
    job = {}
    try:
        logger.info("Scheduling sender job.")
        job = schedule.every(60).minutes.do(send_prepared_requests, kwargs)
    except Exception as e:
        logger.error(e, exc_info=True)

    job.run()
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-send_only', help='Do not prepare new request. Send request from db only.',
                        action='store_true')
    parser.add_argument('-bulk', help='If true, grib files will be requested for an area specified in '
                                      '--nlat, --slat, --wlon and --elon. Otherwise, coordinates from '
                                      '--coordinate_path will be used', action='store_true')
    parser.add_argument('-fetch_city_coordinates', help='Get coordinates for provided cities', action='store_true')
    parser.add_argument('--city_list', help='Path to SYNOP list with locations names', default=False)
    parser.add_argument('--coordinate_path', help='Path to list of cities with coordinates to fetch synop data for.',
                        default=os.path.join(Path(__file__).parent, 'city_coordinates/coordinates_to_fetch.csv'))
    parser.add_argument('--nlat', help='Optional, use for spatial subset requests 90 to -90. Specifies upper bound of '
                                       'spherical rectangle.', default=None, type=float)
    parser.add_argument('--slat', help='Optional, use for spatial subset requests 90 to -90. Specifies lower bound of '
                                       'spherical rectangle.', default=None, type=float)
    parser.add_argument('--wlon', help='Optional, use for spatial subset requests -180 to 180. Specifies western '
                                       'bound of spherical rectangle.', default=None, type=float)
    parser.add_argument('--elon', help='Optional, use for spatial subset requests -180 to 180. Specifies eastern '
                                       'bound of spherical rectangle.', default=None, type=float)
    parser.add_argument('--start_date', help='Start date GFS', default='2018-01-01 00:00')
    parser.add_argument('--end_date', help='End date GFS', default='2019-01-01 00:00')
    parser.add_argument('--input_file', help='Path to JSON input file with parameters, levels and forecast hours type.',
                        type=str, default=None)
    parser.add_argument('--gfs_parameter', help='Parameter to process from NCAR', type=str, default='V GRD')
    parser.add_argument('--gfs_level', help='Level of parameter', type=str, default='HTGL:10')
    parser.add_argument('--forecast_start', help='Offset (in hours) of beginning of the forecast. Should be divisible '
                                                 'by 3.', type=int, default=0)
    parser.add_argument('--forecast_end', help='Offset (in hours) of end of the forecast. Should be divisible by 3.',
                        type=int,
                        default=120)
    parser.add_argument('--hours_type', type=str, choices=['point', 'average', 'all'], help='For some params only 3h '
                                                                                            'averages are available instead of exact time-point forecasts. '
                                                                                            'Set to "average" if you want to fetch dates like "3-hour Average'
                                                                                            ' (initial+0, initial+3)". Set to "all" to fetch both types. '
                                                                                            'Leave empty to use time-points.',
                        default='point')

    args = parser.parse_args()
    os.makedirs(Path(REQ_ID_PATH).parent, exist_ok=True)
    prepare_and_start_processor(**vars(args))

import pandas as pd
import schedule
from rda_request_sender import RequestStatus, REQ_ID_PATH
import rdams_client as rc
from modified_rda_download_client import download
from own_logger import logger
import os, tarfile, re, argparse
import datetime
import sys
import time

sys.path.insert(0, '..')

from utils import prep_zeros_if_needed

RAW_CSV_FILENAME_REGEX = r'(gfs\.0p25\.\d{10}\.f\d{3}\.grib2\.[a-zA-Z]+)(\d+)(.gp.csv)'
RDA_CSV_FILENAME_FORMAT = 'gfs.0p25.{0}{1}{2}{3}.f{4}.grib2.{5}.gp.csv'
FINAL_CSV_FILENAME_FORMAT = '{0}-{1}-{2}-{3}Z.csv'


def read_pseudo_rda_request_db():
    return pd.read_csv(REQ_ID_PATH, index_col=0)


def get_target_path(latitude: str, longitute: str, param: str, level: str):
    latitude = latitude.replace('.', '_')
    longitute = longitute.replace('.', '_')

    return os.path.join(latitude + "-" + longitute, param, level)


def get_download_target_path_tar(req_id: str, latitude: str, longitute: str, param: str, level: str):
    return os.path.join("download/tar", req_id, get_target_path(latitude, longitute, param, level))


def get_download_target_path_csv(latitude: str, longitute: str, param: str, level: str):
    return os.path.join("download/csv", get_target_path(latitude, longitute, param, level))


def download_request(req_id: str, target_dir):
    logger.info("Downloading files from request {0} into {1}".format(req_id, target_dir))
    try:
        download(req_id, target_dir)
    except Exception as e:
        logger.error("Downloading failed", exc_info=True)
        raise e


def purge(req_id: str):
    logger.info("Purging request {}".format(req_id))
    rc.purge_request(req_id)


def print_db_stats(db):
    not_completed = db[db["status"] == RequestStatus.SENT.value]
    logger.info("{} requests are pending".format(len(not_completed)))
    completed = db[db["status"] == RequestStatus.COMPLETED.value]
    logger.info("{} requests are completed".format(len(completed)))
    downloaded = db[db["status"] == RequestStatus.DOWNLOADED.value]
    logger.info("{} requests are downloaded, but not processed yet".format(len(downloaded)))
    finished = db[db["status"] == RequestStatus.FINISHED.value]
    logger.info("{} requests are already processed".format(len(finished)))
    failed = db[db["status"] == RequestStatus.FAILED.value]
    logger.info("{} requests have failed".format(len(failed)))


def extract_files(download_target_path, extract_target_path):
    tars = [f for f in os.listdir(download_target_path) if
            os.path.isfile(os.path.join(download_target_path, f)) and f.endswith("tar")]
    logger.info("Unpacking {0} tars into {1} directory".format(len(tars), extract_target_path))
    for file in tars:
        tar = tarfile.open(os.path.join(download_target_path, file), "r:")
        tar.extractall(extract_target_path)
        tar.close()

    new_csvs_pattern = re.compile(RAW_CSV_FILENAME_REGEX)
    for file in [f for f in os.listdir(extract_target_path) if new_csvs_pattern.match(f)]:
        final_csv_name = re.sub(RAW_CSV_FILENAME_REGEX, r"\1\3", file)
        os.replace(os.path.join(extract_target_path, file), os.path.join(extract_target_path, final_csv_name))


# Update request status by reaching rda web service
def check_request_actual_status(index_in_db, request, request_db, purge_failed):
    req_id = int(request['req_id'])
    res = rc.get_status(req_id)

    if res['status'] == 'Error':
        logger.info("Request id: {} has failed".format(req_id))
        request_db.loc[index_in_db, "status"] = RequestStatus.FAILED.value
        if purge_failed:
            purge(str(req_id))

    elif res['status'] == 'ok':
        request_status = res['result']['status']
        logger.info("Status of request {0} is {1}".format(req_id, request_status))
        if request_status == RequestStatus.ERROR.value:
            request_db.loc[index_in_db, "status"] = RequestStatus.ERROR.value
            if purge_failed:
                purge(str(req_id))
            print(res)

        if request_status == RequestStatus.COMPLETED.value:
            request_db.loc[index_in_db, "status"] = RequestStatus.COMPLETED.value

    else:
        logger.error("Unhandled request status: {0} for request {1}".format(res['status'], req_id))
    request_db.to_csv(REQ_ID_PATH)


def download_completed_request(index_in_db, request, request_db, purge_downloaded: bool):
    req_id = str(int(request['req_id']))
    latitude, longitude = str(request["latitude"]), str(request["longitude"])
    param, level = str(request["param"]), str(request["level"])
    download_target_path = get_download_target_path_tar(req_id, latitude, longitude, param, level)
    os.makedirs(download_target_path, exist_ok=True)
    try:
        download_request(req_id, download_target_path)
        request_db.loc[index_in_db, "status"] = RequestStatus.DOWNLOADED.value
        request_db.to_csv(REQ_ID_PATH)
        if purge_downloaded:
            purge(req_id)
    except Exception as e:
        logger.error(e, exc_info=True)


def process_tars(index_in_db, request, request_db):
    latitude, longitude = str(request["latitude"]), str(request["longitude"])
    param, level = str(request["param"]), str(request["level"])
    download_target_path = get_download_target_path_tar(str(int(request['req_id'])), latitude, longitude, param, level)
    extract_target_path = get_download_target_path_csv(latitude, longitude, param, level)
    extract_files(download_target_path, extract_target_path)
    request_db.loc[index_in_db, "status"] = RequestStatus.FINISHED.value


def get_value_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, error_bad_lines=False, warn_bad_lines=False,
                     names=['#ParameterName', 'Longitude', 'Latitude', 'ValidDate', 'ValidTime', 'LevelValue',
                            'ParameterValue'])
    value = df.iloc[1]['ParameterValue']
    if value == 'lev':  # in some csvs first row is different
        value = df.iloc[2]['ParameterValue']
    return value


def prepare_final_csvs(dir_with_csvs, latitude: str, longitude: str):
    final_date = datetime.datetime.now()
    latlon_dir = os.path.join("download/csv", dir_with_csvs)
    final_csv_dir = os.path.join("output_csvs", latitude.replace('.', '_') + '-' + longitude.replace('.', '_'))
    if not os.path.exists(final_csv_dir):
        os.makedirs(final_csv_dir)

    init_date = datetime.datetime(2015, 1, 15)
    while init_date < final_date:
        for run in ['00', '06', '12', '18']:
            # final csv name for this forecast
            final_csv_name = FINAL_CSV_FILENAME_FORMAT.format(init_date.year,
                                                              prep_zeros_if_needed(str(init_date.month), 1),
                                                              prep_zeros_if_needed(str(init_date.day), 1),
                                                              run)
            final_csv_path = os.path.join(final_csv_dir, final_csv_name)
            if os.path.exists(final_csv_path):
                forecast_df = pd.read_csv(final_csv_path, index_col=[0])
            else:
                forecast_df = pd.DataFrame(columns=["date"])
            offset = 3
            while offset < 168:
                # fetch values from all params and levels
                for root, param_dirs, filenames in os.walk(latlon_dir):
                    for param_dir in param_dirs:
                        for root, level_dirs, filenames in os.walk(os.path.join(latlon_dir, param_dir)):
                            for level_dir in level_dirs:
                                level_dir_path = os.path.join(latlon_dir, param_dir, level_dir)
                                csv_file_name = RDA_CSV_FILENAME_FORMAT.format(init_date.year, prep_zeros_if_needed(str(init_date.month), 1),
                                                                               prep_zeros_if_needed(str(init_date.day),1),
                                                                               run,
                                                                               prep_zeros_if_needed(str(offset), 2),
                                                                               'belniak')
                                csv_file_path = os.path.join(level_dir_path, csv_file_name)

                                if os.path.exists(csv_file_path):
                                    value = get_value_from_csv(csv_file_path)
                                    param_name = param_dir + '_' + level_dir
                                    date = (init_date + datetime.timedelta(hours=offset)).isoformat()

                                    if param_name not in forecast_df:
                                        forecast_df[param_name] = 0

                                    rows = forecast_df[forecast_df['date'] == date]
                                    if len(rows) == 0:
                                        forecast_df.loc[len(forecast_df.index), :] = ''
                                        forecast_df.loc[len(forecast_df.index) - 1, ['date', param_name]] = [date, value]
                                        forecast_df.sort_values('date')
                                    else:
                                        forecast_df.loc[forecast_df['date'] == date, param_name] = value
                                    os.remove(csv_file_path)

                            break  # check only first-level subdirectories
                    break  # check only first-level subdirectories

                offset = offset + 3
            forecast_df.to_csv(final_csv_path)
        init_date = init_date + datetime.timedelta(days=1)


def processor(purge: bool):
    logger.info("Starting rda download processor")
    request_db = read_pseudo_rda_request_db()
    print_db_stats(request_db)
    print("Checking actual status of pending requests...")

    not_completed = request_db[request_db["status"] == RequestStatus.SENT.value]

    for index, request in not_completed.iterrows():
        check_request_actual_status(index, request, request_db, purge)

    completed = request_db[request_db["status"] == RequestStatus.COMPLETED.value]
    for index, request in completed.iterrows():
        download_completed_request(index, request, request_db, purge)

    ready_for_unpacking = request_db[request_db["status"] == RequestStatus.DOWNLOADED.value]
    for index, request in ready_for_unpacking.iterrows():
        process_tars(index, request, request_db)

    print("Processing csv files...")
    for root, dirs, filenames in os.walk("download/csv"):
        for dir_with_csvs in dirs:
            latlon_search = re.search(r'(\d+(_\d)?)-(\d+(_\d)?)', dir_with_csvs)
            latitude = latlon_search.group(1)
            longitude = latlon_search.group(3)
            prepare_final_csvs(dir_with_csvs, latitude, longitude)
        break

    request_db.to_csv(REQ_ID_PATH)
    print("Done. Waiting for next scheduler trigger.")


def scheduler(purge=False):
    try:
        job = schedule.every(15).minutes.do(lambda: processor(purge))
    except Exception as e:
        logger.error(e, exc_info=True)

    job.run()
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--purge', help='Purge request after downloading files or if the request has failed',
                        default=False)

    args = parser.parse_args()
    scheduler(args.purge)

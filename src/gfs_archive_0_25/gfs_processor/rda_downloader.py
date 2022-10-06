import numpy
import pandas as pd
import schedule
import tarfile, re, argparse
import time

from gfs_archive_0_25.gfs_processor.rda_request_sender import RequestStatus, REQ_ID_PATH, RequestType
import gfs_archive_0_25.gfs_processor.rdams_client as rc
from gfs_archive_0_25.gfs_processor.modified_rda_download_client import download
from gfs_archive_0_25.gfs_processor.own_logger import logger
from gfs_archive_0_25.gfs_processor.consts import *


def read_pseudo_rda_request_db():
    return pd.read_csv(REQ_ID_PATH, index_col=0)


def get_download_target_path_tar(request_id: str, request_type: str, nlat: str, elon: str, param: str, level: str):
    if request_type == RequestType.POINT.value:
        lat = nlat.replace('.', '_')
        lon = elon.replace('.', '_')
        return os.path.join(TAR_DOWNLOAD_PATH, request_id, lat + "-" + lon, param, level.replace(":", "_").replace(',', '-'))
    else:
        return os.path.join(TAR_DOWNLOAD_PATH, request_id, param, level.replace(":", "_").replace(',', '-'))


def get_unpacked_target_path(request_type: str, nlat: str, elon: str, param: str, level: str):
    if request_type == RequestType.POINT.value:
        lat = nlat.replace('.', '_')
        lon = elon.replace('.', '_')
        return os.path.join(CSV_DOWNLOAD_PATH, lat + "-" + lon, param, level.replace(":", "_").replace(",", "-"))
    else:
        return os.path.join(NETCDF_DOWNLOAD_PATH, param, level.replace(":", "_").replace(',', '-'))


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
    not_completed = db[db[REQUEST_STATUS_FIELD] == RequestStatus.SENT.value]
    logger.info("{} requests are pending".format(len(not_completed)))
    completed = db[db[REQUEST_STATUS_FIELD] == RequestStatus.COMPLETED.value]
    logger.info("{} requests are completed".format(len(completed)))
    downloaded = db[db[REQUEST_STATUS_FIELD] == RequestStatus.DOWNLOADED.value]
    logger.info("{} requests are downloaded, but not processed yet".format(len(downloaded)))
    finished = db[db[REQUEST_STATUS_FIELD] == RequestStatus.FINISHED.value]
    logger.info("{} requests are already processed".format(len(finished)))
    failed = db[db[REQUEST_STATUS_FIELD] == RequestStatus.FAILED.value]
    logger.info("{} requests have failed".format(len(failed)))


def extract_files_from_tar(download_target_path, extract_target_path, file_type: str, tidy=False):
    tars = [f for f in os.listdir(download_target_path) if
            os.path.isfile(os.path.join(download_target_path, f)) and f.endswith("tar")]
    logger.info("Unpacking {0} tars into {1} directory".format(len(tars), extract_target_path))
    for file in tars:
        tar = tarfile.open(os.path.join(download_target_path, file), "r:")
        tar.extractall(extract_target_path)
        tar.close()
        if tidy:
            os.remove(tar)

    if file_type == "csv":
        new_file_pattern = re.compile(RAW_CSV_FILENAME_WITH_REQUEST_REGEX)
        for file in [f for f in os.listdir(extract_target_path) if new_file_pattern.match(f)]:
            final_csv_name = re.sub(RAW_CSV_FILENAME_WITH_REQUEST_REGEX, r"\1\5", file)  # remove request number
            os.replace(os.path.join(extract_target_path, file), os.path.join(extract_target_path, final_csv_name))
    else:
        new_file_pattern = re.compile(RAW_NETCDF_FILENAME_WITH_REQUEST_REGEX)
        for file in [f for f in os.listdir(extract_target_path) if new_file_pattern.match(f)]:
            final_csv_name = re.sub(RAW_NETCDF_FILENAME_WITH_REQUEST_REGEX, r"\1\5", file)
            os.replace(os.path.join(extract_target_path, file), os.path.join(extract_target_path, final_csv_name))


# Update request status by reaching rda web service
def check_request_actual_status(index_in_db, request, request_db):
    req_id = int(request[REQUEST_ID_FIELD])
    res = rc.get_status(req_id)

    if res['status'].lower() == 'error':
        logger.info("Request file_id: {} has failed".format(req_id))
        request_db.loc[index_in_db, REQUEST_STATUS_FIELD] = RequestStatus.FAILED.value

    elif res['status'] == 'ok':
        request_status = res['result']['status']
        logger.info("Status of request {0} is {1}".format(req_id, request_status))
        if request_status == RequestStatus.ERROR.value:
            request_db.loc[index_in_db, REQUEST_STATUS_FIELD] = RequestStatus.ERROR.value
            logger.info(res)

        if request_status == RequestStatus.COMPLETED.value:
            request_db.loc[index_in_db, REQUEST_STATUS_FIELD] = RequestStatus.COMPLETED.value

    else:
        logger.error("Unhandled request status: {0} for request {1}".format(res['status'], req_id))
    request_db.to_csv(REQ_ID_PATH)


def download_completed_request(index_in_db, request, request_db):
    req_id, req_type = str(int(request[REQUEST_ID_FIELD])), str(request[REQUEST_TYPE_FIELD])
    nlat, elon = str(request[NLAT_FIELD]), str(request[ELON_FIELD])
    param, level = str(request[PARAM_FIELD]), str(request[LEVEL_FIELD])
    download_target_path = get_download_target_path_tar(req_id, req_type, nlat, elon, param, level)
    os.makedirs(download_target_path, exist_ok=True)
    try:
        download_request(req_id, download_target_path)
        request_db.loc[index_in_db, REQUEST_STATUS_FIELD] = RequestStatus.DOWNLOADED.value
        request_db.to_csv(REQ_ID_PATH)

    except Exception as e:
        logger.error(e, exc_info=True)


def process_tars(index_in_db, request, request_db, tidy: bool):
    nlat, elon = str(request[NLAT_FIELD]), str(request[ELON_FIELD])
    request_id, request_type = str(int(request[REQUEST_ID_FIELD])), str(request[REQUEST_TYPE_FIELD])
    param, level = str(request[PARAM_FIELD]), str(request[LEVEL_FIELD])
    download_target_path = get_download_target_path_tar(request_id, request_type, nlat, elon, param, level)
    extract_target_path = get_unpacked_target_path(request_type, nlat, elon, param, level)
    extract_files_from_tar(download_target_path, extract_target_path, "csv" if request_type == RequestType.POINT.value else "netCDF", tidy)
    request_db.loc[index_in_db, REQUEST_STATUS_FIELD] = RequestStatus.FINISHED.value


def processor(purge_requests: bool, tidy: bool):
    logger.info("Starting rda download processor")
    request_db = read_pseudo_rda_request_db()
    print_db_stats(request_db)
    logger.info("Checking actual status of pending requests...")

    not_completed = request_db[request_db[REQUEST_STATUS_FIELD] == RequestStatus.SENT.value]

    for index, request in not_completed.iterrows():
        check_request_actual_status(index, request, request_db)

    completed = request_db[request_db[REQUEST_STATUS_FIELD] == RequestStatus.COMPLETED.value]
    for index, request in completed.iterrows():
        download_completed_request(index, request, request_db)

    ready_for_unpacking = request_db[request_db[REQUEST_STATUS_FIELD] == RequestStatus.DOWNLOADED.value]
    for index, request in ready_for_unpacking.iterrows():
        process_tars(index, request, request_db, tidy)

    request_db.to_csv(REQ_ID_PATH)

    if purge_requests:
        done_requests = request_db[(request_db[REQUEST_STATUS_FIELD] == RequestStatus.ERROR.value) |
                                   (request_db[REQUEST_STATUS_FIELD] == RequestStatus.FAILED.value) |
                                   (request_db[REQUEST_STATUS_FIELD] == RequestStatus.DOWNLOADED.value) |
                                   (request_db[REQUEST_STATUS_FIELD] == RequestStatus.FINISHED.value)]
        for index, request in done_requests.iterrows():
            if not numpy.isnan(request[REQUEST_ID_FIELD]):
                purge(str(int(request[REQUEST_ID_FIELD])))
                request_db.loc[index, REQUEST_STATUS_FIELD] = RequestStatus.PURGED.value
        request_db.to_csv(REQ_ID_PATH)

    logger.info("Done. Waiting for next scheduler trigger.")


def scheduler(purge_done=False, tidy=False):
    try:
        job = schedule.every(60).minutes.do(lambda: processor(purge_done, tidy))
        job.run()
        while True:
            schedule.run_pending()
            time.sleep(60)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-purge', help='Purge request after downloading files or if the request has failed',
                        default=False, action="store_true")
    parser.add_argument('-tidy', help='Remove downloaded tars after unpacking.',
                        default=False, action="store_true")

    args = parser.parse_args()
    scheduler(args.purge, args.tidy)

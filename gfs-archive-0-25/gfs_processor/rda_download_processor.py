import pandas as pd
import schedule
from rda_request_sender import RequestStatus, REQ_ID_PATH
import rdams_client as rc
from modified_rda_download_client import download
from own_logger import logger
import os, tarfile


def read_pseudo_rda_request_db():
    return pd.read_csv(REQ_ID_PATH, index_col=0)


def get_target_path(req_id: str, latitude: str, longitute: str, param: str, level: str):
    latitude = latitude.replace('.', '_')
    longitute = longitute.replace('.', '_')

    return os.path.join(latitude + "_" + longitute, param, level, req_id)


def get_download_target_path_tar(req_id: str, latitude: str, longitute: str, param: str, level: str):
    return os.path.join("download/tar", get_target_path(req_id, latitude, longitute, param, level))


def get_download_target_path_csv(req_id: str, latitude: str, longitute: str, param: str, level: str):
    return os.path.join("download/csv", get_target_path(req_id, latitude, longitute, param, level))


def download_request(req_id: int, target_dir):
    logger.info("Downloading files from request {0} into {1}".format(req_id, target_dir))
    try:
        download(req_id, target_dir)
    except Exception as e:
        logger.error("Downloading failed", exc_info=True)
        raise e


def purge(req_id: int):
    rc.purge_request(req_id)


def print_db_stats(db):
    not_completed = db[db["status"] == RequestStatus.SENT.value]
    logger.info("{} requests are pending".format(len(not_completed)))
    completed = db[db["status"] == RequestStatus.COMPLETED.value]
    logger.info("{} requests are completed".format(len(completed)))
    failed = db[db["status"] == RequestStatus.FAILED.value]
    logger.info("{} requests have failed".format(len(failed)))


def extract_files(download_target_path, extract_target_path):
    tars = [f for f in os.listdir(download_target_path) if os.path.isfile(os.path.join(download_target_path, f)) and f.endswith("tar")]
    logger.info("Unpacking {0} tars into {1} directory".format(len(tars), extract_target_path))
    for file in tars:
        tar = tarfile.open(os.path.join(download_target_path, file), "r:")
        tar.extractall(extract_target_path)
        tar.close()


# Update request status by reaching rda web service
def check_request_actual_status(index_in_db, request, request_db):
    req_id = request['req_id']
    res = rc.get_status(req_id)

    if res['status'] == 'error':
        logger.info("Request id: {} has failed".format(req_id))
        request_db.loc[index_in_db, "status"] = RequestStatus.FAILED.value

    elif res['status'] == 'ok':
        request_status = res['result']['status']
        logger.info("Status of request {0} is {1}".format(req_id, request_status))

        if request_status == RequestStatus.COMPLETED.value:
            request_db.loc[index_in_db, "status"] = RequestStatus.COMPLETED.value
            
    else:
        logger.error("Unhandled request status: {}".format(res['status']))


def download_completed_request(index_in_db, request, request_db):
    req_id = request['req_id']
    latitude, longitude = str(request["latitude"]), str(request["longitude"])
    param, level = str(request["param"]), str(request["level"])
    download_target_path = get_download_target_path_tar(str(req_id), latitude, longitude, param, level)
    os.makedirs(download_target_path, exist_ok=True)
    try:
        download_request(req_id, download_target_path)
        request_db.loc[index_in_db, "status"] = RequestStatus.DOWNLOADED.value
        # purge(req_id)
    except Exception as e:
        logger.error(e, exc_info=True)


def process_tars(index_in_db, request, request_db):
    req_id = request['req_id']
    latitude, longitude = str(request["latitude"]), str(request["longitude"])
    param, level = str(request["param"]), str(request["level"])
    download_target_path = get_download_target_path_tar(str(req_id), latitude, longitude, param, level)
    extract_target_path = get_download_target_path_csv(str(req_id), latitude, longitude, param, level)
    extract_files(download_target_path, extract_target_path)
    request_db.loc[index_in_db, "status"] = RequestStatus.FINISHED.value


def processor():
    logger.info("Starting rda download processor")
    request_db = read_pseudo_rda_request_db()
    print_db_stats(request_db)

    not_completed = request_db[request_db["status"] == RequestStatus.SENT.value]
    
    for index, request in not_completed.iterrows():
        check_request_actual_status(index, request, request_db)
    
    completed = request_db[request_db["status"] == RequestStatus.COMPLETED.value]
    for index, request in completed.iterrows():
        download_completed_request(index, request, request_db)
        
    ready_for_unpacking = request_db[request_db["status"] == RequestStatus.DOWNLOADED.value]
    for index, request in ready_for_unpacking.iterrows():
        process_tars(index, request, request_db)
    
    request_db.to_csv(REQ_ID_PATH)


def scheduler():
    try:
        schedule.every(3).hours.do(processor)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    processor()

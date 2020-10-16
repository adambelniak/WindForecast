import pandas as pd
import schedule
from rda_request_sender import RequestStatus, REQ_ID_PATH
import rdams_client as rc
from modified_rda_download_client import download
from own_logger import logger
import os


def read_pseudo_rda_request_db():
    return pd.read_csv(REQ_ID_PATH, index_col=0)


def get_download_target_path(req_id: str, latitude: str, longitute: str, param: str, level: str):
    latitude = latitude.replace('.', '_')
    longitute = longitute.replace('.', '_')

    return os.path.join("download/tar", latitude + "_" + longitute, param, level, req_id)


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


def processor():
    logger.info("Starting rda download processor")
    request_db = read_pseudo_rda_request_db()
    print_db_stats(request_db)

    not_completed = request_db[request_db["status"] == RequestStatus.SENT.value]
    for index, request in not_completed.iterrows():
        req_id = request['req_id']
        res = rc.get_status(req_id)

        if res['status'] == 'error':
            logger.info("Request id: {} has failed".format(req_id))
            request_db.loc[index, "status"] = RequestStatus.FAILED.value

        elif res['status'] == 'ok':
            request_status = res['result']['status']
            logger.info("Status of request {0} is {1}".format(req_id, request_status))

            if request_status == RequestStatus.COMPLETED.value:
                latitude, longitude = str(request["latitude"]), str(request["longitude"])
                param, level = str(request["param"]), str(request["level"])
                download_target_path = get_download_target_path(str(req_id), latitude, longitude, param, level)
                os.makedirs(download_target_path, exist_ok=True)
                try:
                    download_request(req_id, download_target_path)
                    request_db.loc[index, "status"] = RequestStatus.COMPLETED.value
                    #purge(req_id)
                except Exception as e:
                    logger.error(e, exc_info=True)
        else:
            logger.error("Unhandled request status: {}".format(res['status']))

    request_db.to_csv(REQ_ID_PATH)


def scheduler():
    try:
        schedule.every(3).hours.do(processor)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    processor()

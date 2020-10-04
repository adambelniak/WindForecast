import pandas as pd
import schedule
from rda_request_sender import RequestStatus, REQ_ID_PATH
import rdams_client as rc
from modified_rda_download_client import download
from own_logger import logger
from pathlib import Path
import os


def read_pseudo_rda_request_db():
    return pd.read_csv(REQ_ID_PATH, index_col=0)


def create_dir_by_location_and_request(req_id: str, latitude: str, longitute: str):
    latitude = latitude.replace('.', '_')
    longitute = longitute.replace('.', '_')

    location_path = latitude + "_" + longitute
    Path(location_path).mkdir(parents=True, exist_ok=True)
    request_path = os.path.join(location_path, req_id)
    Path(request_path).mkdir(parents=True, exist_ok=True)
    return location_path, request_path


def download_request(req_id: int, latitude: str, longitute: str):
    _, request_path = create_dir_by_location_and_request(str(req_id), latitude, longitute)
    logger.info("start downloading")
    try:
        download(req_id, request_path)
    except Exception as e:
        logger.error("Downloading failed", exc_info=True)
        raise e


def purge(req_id: int):
    rc.purge_request(req_id)


def processor():
    logger.info("Start processor")
    request_db = read_pseudo_rda_request_db()
    not_completed = request_db[request_db["status"] == RequestStatus.SENT.value]
    logger.info("{} requests is pending".format(len(not_completed)))
    for index, request in not_completed.iterrows():
        req_id = request['req_id']
        res = rc.get_status(req_id)
        request_status = res['result']['status']

        if request_status == 'Completed':
            logger.info("Request id: {} is completed".format(req_id))
            try:
                download_request(req_id, str(request["latitude"]), str(request["longitude"]))
                request_db.loc[index,"status"] = RequestStatus.COMPLETED.value
                purge(req_id)
            except Exception as e:
                logger.error(e, exc_info=True)

    request_db.to_csv(REQ_ID_PATH)


def scheduler():
    try:
        schedule.every(3).hours.do(processor)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == '__main__':
    processor()

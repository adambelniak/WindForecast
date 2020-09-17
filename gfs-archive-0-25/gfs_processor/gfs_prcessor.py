import sys

import pandas as pd
import schedule
from rda_request_sender import RequestStatus, REQ_ID_PATH
import rdams_client as rc
import logging


def set_logger():
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    handler = logging.FileHandler('gfs_prcessor.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


logger = set_logger()


def read_pseudo_rda_request_db():
    return pd.read_csv(REQ_ID_PATH, index_col=0)


def download_request(req_id):
    logger.info("start downloading")
    try:
        rc.download(req_id)
    except Exception as e:
        logger.error("Downloading failed")
        raise e


def check_rda_request_status():
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
                download_request(req_id)
                request_db.at[index]["status"] = RequestStatus.COMPLETED.value
            except Exception as e:
                logger.error(e)

    request_db.to_csv(REQ_ID_PATH)


def processor():
    pass


def scheduler():
    schedule.every(3).hours.do(processor)


if __name__ == '__main__':
    check_rda_request_status()

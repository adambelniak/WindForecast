import os
import requests
import sys
sys.path.insert(0, '../rda-apps-clients/src/python')
sys.path.insert(1, '..')
import rdams_client as rc
from own_logger import logger
import subprocess

BASE_URL = 'https://rda.ucar.edu/json_apps/'
USE_NETRC = False
DEFAULT_AUTH_FILE = './rdamspw.txt'


def get_filelist(request_idx):
    """Gets filelist for request
    Args:
        request_idx (str): Request Index, typically a 6-digit integer.
    Returns:
        dict: JSON decoded result of the query.
    """
    base = 'https://rda.ucar.edu/apps/'

    url = base + 'request/'
    url += str(request_idx)
    url += '/filelist'

    user_auth = rc.get_authentication()
    ret = requests.get(url, auth=user_auth)
    return ret


def download_files(filelist, out_dir='./', cookie_file=None):
    """Download files in a list.
    Args:
        filelist (list): List of web files to download.
        out_dir (str): directory to put downloaded files
    Returns:
        None
    """
    if cookie_file is None:
        cookies = rc.get_cookies()
    for _file in filelist:
        file_base = os.path.basename(_file)
        out_file = os.path.join(out_dir, file_base)
        logger.info('Downloading file {}'.format(file_base))
        req = requests.get(_file, cookies=cookies, allow_redirects=True, stream=True)
        filesize = int(req.headers['Content-length'])
        with open(out_file, 'wb') as outfile:
            chunk_size=1048576
            for chunk in req.iter_content(chunk_size=chunk_size):
                outfile.write(chunk)
                if chunk_size < filesize:
                    rc.check_file_status(out_file, filesize)
        rc.check_file_status(out_file, filesize)
        logger.info('')


def download(request_idx, out_dir):
    """Download files given request Index
    Args:
        request_idx (str): Request Index, typically a 6-digit integer.
    Returns:
        None
    """
    ret = get_filelist(request_idx)
    if ret.status_code != 200:
        return ret

    filelist = ret.json()

    user_auth = rc.get_authentication()

    username, password = user_auth
    cookies = rc.get_cookies(username,password)

    web_files = list(map(lambda x: x, filelist.keys()))

    download_files(web_files, out_dir)
    return ret

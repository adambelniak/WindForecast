from __future__ import annotations
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import requests

from gfs_archive_0_25.utils import prep_zeros_if_needed
from gfs_oper.common import InitMeta, InitHour, Config
from util.coords import Coords
from util.util import download

NCEP_URL_TEMPLATE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{0}/{1}/atmos/gfs.t{1}z.pgrb2.0p25.f{2}"


class GribResource:
    init_meta: InitMeta
    offset: int

    def __init__(self, init_meta: InitMeta, offset: int) -> None:
        super().__init__()
        self.init_meta = init_meta
        self.offset = offset

    def fetch(self, output_path: str) -> None:
        output_location = self.get_output_location(output_path)
        if os.path.exists(output_location):
            return
        os.makedirs(os.path.dirname(output_location), exist_ok=True)
        if not self.is_available():
            raise ResourceUnavailableException(self.url())
        download(output_location, self.url())

    def url(self) -> str:
        return NCEP_URL_TEMPLATE.format(self.init_meta.get_init_date_string(),
                                        self.init_meta.init_hour.value,
                                        prep_zeros_if_needed(str(self.offset), 2))

    def is_available(self) -> bool:
        return requests.head(self.url()).ok

    def get_output_location(self, output_path):
        output_location = os.path.join(output_path, self.init_meta.get_init_date_string(),
                                       str(self.init_meta.init_hour.value),
                                       os.path.basename(self.url()))
        return os.path.join(Path(__file__).parent, output_location)


class ResourceUnavailableException(Exception):
    url: str = ...

    def __init__(self, url: str, *args: object) -> None:
        super().__init__(*args)
        self.url = url

    def __str__(self):
        return f"Resource unavailable at url {self.url}"


def get_init_meta(date: datetime) -> InitMeta:
    utc_date = datetime.utcfromtimestamp(datetime.timestamp(date))
    hour = utc_date.hour
    init_hour = InitHour.get_init_hour(date)
    if init_hour.value == InitHour._18z.value and hour < 5:
        utc_date = utc_date - timedelta(days=1)
    return InitMeta(utc_date, init_hour)

def get_init_meta_for_run(date: datetime, init_hour: InitHour) -> InitMeta:
    return InitMeta(date, init_hour)


def get_init_meta_from_init_string(init_string: str) -> InitMeta:
    date_matcher = re.match(r'(\d{4})(\d{2})(\d{2}) (\d{2}):00', init_string)
    date = datetime(year=int(date_matcher.group(1)),
                    month=int(date_matcher.group(2)),
                    day=int(date_matcher.group(3)))
    init_hour = InitHour(prep_zeros_if_needed(date_matcher.group(4), 1))
    return get_init_meta_for_run(date, init_hour)


def fetch_future_gribs(init_meta: InitMeta, future_sequence_length: int, output_path: str) -> None:
    for offset in range(0, future_sequence_length):
        GribResource(init_meta, offset).fetch(output_path)


def fetch_past_gribs(init_meta: InitMeta, past_sequence_length: int, output_path: str) -> None:
    for init in range(0, (past_sequence_length // 6) + 1):
        init_meta = init_meta.get_previous()
        for offset in range(0, 6):
            GribResource(init_meta, offset).fetch(output_path)


def fetch_all_needed_gribs(init_meta: InitMeta, config: Config):
    fetch_future_gribs(init_meta, config.future_sequence_length, config.download_path)
    fetch_past_gribs(init_meta, config.past_sequence_length, config.download_path)


def fetch_recent_gfs(config: Config) -> bool:
    current_date = datetime.now()
    init_meta = get_init_meta(current_date)

    forecast_available = False
    tries = 0

    while not forecast_available and tries < 5:
        grib_resource = GribResource(init_meta, config.future_sequence_length)
        forecast_available = grib_resource.is_available()
        if not forecast_available:
            current_date = current_date - timedelta(hours=6)
            init_meta = init_meta.get_previous()
            tries += 1

    if forecast_available:
        fetch_all_needed_gribs(init_meta, config)

    return forecast_available


if __name__ == "__main__":
    fetch_recent_gfs(Config(24, 1, "download", "processed", Coords(52.1831174, 52.1831174, 20.9875259, 20.9875259)))

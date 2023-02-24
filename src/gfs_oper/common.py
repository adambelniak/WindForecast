from __future__ import annotations
from datetime import datetime
from enum import Enum

from util.coords import Coords


class Config:
    past_sequence_length: int = ...
    future_sequence_length: int = ...
    download_path: str = ...
    processing_output_path: str = ...
    target_coords: Coords = ...

    def __init__(self, past_sequence_length: int, future_sequence_length: int, download_path: str, processing_output_path: str,
                 target_coords: Coords) -> None:
        super().__init__()
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.download_path = download_path
        self.processing_output_path = processing_output_path
        self.target_coords = target_coords


class InitHour(Enum):
    _00z = '00'
    _06z = '06'
    _12z = '12'
    _18z = '18'

    @staticmethod
    def get_init_hour(date: datetime):
        utc_date = datetime.utcfromtimestamp(datetime.timestamp(date))
        hour = utc_date.hour

        if 5 <= hour < 11:
            return InitHour._00z
        if 11 <= hour < 17:
            return InitHour._06z
        if 17 <= hour < 23:
            return InitHour._12z
        return InitHour._18z

    def get_previous(self) -> InitHour:
        if self.value == InitHour._00z.value:
            return InitHour._18z
        if self.value == InitHour._06z.value:
            return InitHour._00z
        if self.value == InitHour._12z.value:
            return InitHour._06z
        return InitHour._12z


class InitMeta:
    date: datetime = ...
    init_hour: InitHour = ...

    def __init__(self, date: datetime, init_hour: InitHour) -> None:
        super().__init__()
        self.date = date
        self.init_hour = init_hour

    def get_init_date_string(self):
        return self.date.strftime('%Y%m%d')
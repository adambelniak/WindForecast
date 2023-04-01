from dataclasses import dataclass


@dataclass
class Coords:

    def __init__(self, nlat, slat, wlon, elon):
        self.nlat = nlat
        self.slat = slat
        self.wlon = wlon
        self.elon = elon


POLAND_NLAT = 56
POLAND_SLAT = 48
POLAND_WLON = 13
POLAND_ELON = 26
GFS_SPACE = Coords(POLAND_NLAT, POLAND_SLAT, POLAND_WLON, POLAND_ELON)
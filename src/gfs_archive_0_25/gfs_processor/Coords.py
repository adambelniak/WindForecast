from dataclasses import dataclass


@dataclass
class Coords:

    def __init__(self, nlat, slat, wlon, elon):
        self.nlat = nlat
        self.slat = slat
        self.wlon = wlon
        self.elon = elon

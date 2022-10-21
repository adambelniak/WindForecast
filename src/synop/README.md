# Fetch synop data

Fetch SYNOP data from desired weather station. Source: (IMGW)[https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/]

## Running

In src:
```
python -m synop.fetch_synop_data [...options]
# e.g.
python -m synop.fetch_synop_data --station_code=254180010 --start_year=2015
```

## Configuration
```
  --working_dir WORKING_DIR Working directory. Files will be saved there in synop_data and auto_station_data dirs
  --station_code STATION_CODE
                        Localisation code for which to get data from stations. Codes can be found at https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/wykaz_stacji.csv
  --localisation_name LOCALISATION_NAME
                        Localisation name for which to get data. Will be used to generate final files name.
  --start_year START_YEAR
                        Start date for fetching data
  --end_year END_YEAR   End date for fetching data
```
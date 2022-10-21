# Fetch CMAX radar data

Fetch maximum reflectivity images from Poland area. Source: [https://danepubliczne.imgw.pl/datastore](https://danepubliczne.imgw.pl/datastore).

## Running

In src:
1. Run `python -m radar.fetch_radar_cmax`
2. Run `python -m radar.preprocess_cmax`

## Configuration

Set `CMAX_DATASET_DIR` env variable as the output directory of downloaded files. Otherwise, directory 'data' will be created and used.

Cmd parameters for both scripts:
```
  --from_year FROM_YEAR
                        Start fetching/processing from this year. Must be between 2011 and 2022
  --from_month FROM_MONTH
                        Start fetching/processing from this month
  --to_year TO_YEAR     Fetch/process up to this year
  --to_month TO_MONTH   Fetch/process up to this month
```

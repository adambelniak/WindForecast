Basically, we download netCDF4 files via python script.
Initially, we used subsetting GUI at https://rda.ucar.edu/datasets/ds084.1/index.html#!cgi-bin/datasets/getSubset?dsnum=084.1&listAction=customize&_da=y.
We switched to a python client to automize the process: https://github.com/NCAR/rda-apps-clients/tree/master/src/python
When using the python client, base on their README and our txt/GFSControlFileTemplate.txt

<b>Right now scripts might use hard-coded paths in several places, so if you want to reuse them you may need to modify the source code. To be fixed.</b>

The source code of our flow is placed in gfs_processor module.
There are 3 main files:

#### 1. rda_request_sender.py
Prepares the request body adhering to RDA API (see https://github.com/NCAR/rda-apps-clients/tree/main/src/python) and sends it.  
Requests are prepared based on `--input_file` parameter. You can request for one-point coordinate or a spatial region by switching `-bulk` flag.  
An example of input file: gfs_processor/gfs_params.json. `hours_type` needs to be specified, see help for `--hours_type` CLI argument.
Before being sent, requests metadata is saved in `csv/req_list.csv`. The metadata consists of e.g. request id and request status.

RDA API is limited to 10 requests at a time, so this tool uses scheduler to check frequently if it can send another request if an old one has been purged.

That's an example of how we use rda_request_sender.py:
```
python rda_request_sender.py -bulk --nlat=55.5 --slat=55.25 --elon=13 --wlon=13.25 --input_file=input_files/gfs_params.json --forecast_end=120
```

#### 2. rda_downloader.py
Checks the statuses of requests and downloads the ones that are ready. Then, it untars downloaded files. It uses a scheduler, so it checks every 60 minutes if there are new requests ready.

#### 3. rda_netCDF_files_processor.py
This script is responsible for processing netCDF4 files (coming from point 2.) to either .csv or .npy format for further, easier manipulation.
We use .npy, because later we process them into pickle files, which aggregate forecast from multiple dates.
Each output file consist of one parameter forecast for one time frame.

#### 4. convert_gfs_files.py
Scans GFS_DATASET_DIR for .npy files and for each offset between 3 and 39 and each parameter creates a .pkl file.  
.pkl files are used in experiments by GFSLoader for faster data reading.
For running experiments, GFS_DATASET_DIR should point to a folder containing `pkl` folder with prepared .pkl files.

## Deprecated
### grib_to_csv.py

Names of grib files should match the regex {0}/{0}{1}{2}/gfs.0p25.{0}{1}{2}{3}.f{4}.grib2

usage: grib_to_csv.py [-h] [--dir DIR] [--out OUT] [--lat LAT] [--long LONG] [--shortName SHORTNAME] [--type TYPE] [--level LEVEL]

optional arguments:<br>
```
  -h, --help            show this help message and exit
  --dir DIR             Directory with grib files. Default: current directory. 
  --out OUT             Output directory for csv files. Default: current directory. 
  --lat LAT             Latitude of point for which to get the data. Default: 54.6 
  --long LONG           Longitude of point for which to get the data. Default: 18.8 
  --shortName SHORTNAME
                        Short name of parameter to fetch. Default: 2t 
  --type TYPE           Type of level 
  --level LEVEL         Level 
```

### merge_csv.py
usage: merge_csv.py [-h] --dirs DIRS [DIRS ...] --names NAMES [NAMES ...] [--out OUT]

optional arguments: <br>
```
  -h, --help            show this help message and exit 
  --dirs DIRS [DIRS ...]
                        Input directories with csv files. Names of files should be in format YYYY-mm-dd-HHZ.csv 
  --names NAMES [NAMES ...]
                        Names of the parameters for each input csv 
  --out OUT             Output directory for csv files 
```


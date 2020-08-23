Basically, we don't download grib files via python script.
We use subsetting GUI at https://rda.ucar.edu/datasets/ds084.1/index.html#!cgi-bin/datasets/getSubset?dsnum=084.1&listAction=customize&_da=y

This is the flow we perform:

    1. Request a data subset via the aforementioned link.
    2. When the data is ready, download tar files with python script provided by RDA.
    3. Use grib_to_csv.py to fetch data from grib file and save it to csv files. 
       Each csv file will contain one GFS forecast for one variable.
    4. Use merge_csv.py to merge one-parameter csv files into multi-parameter csv files.
    5. Upload files to drive and use.
    
We request a subset of only one variable, because the pygrib library is much faster 
when there is only one variable in the file. Thus, the need for merge_csv.py script.

# grib_to_csv.py

Names of grib files should match the regex {0}/{0}{1}{2}/gfs.0p25.{0}{1}{2}{3}.f{4}.grib2

usage: grib_to_csv.py [-h] [--dir DIR] [--out OUT] [--lat LAT] [--long LONG] [--shortName SHORTNAME] [--type TYPE] [--level LEVEL]

optional arguments:<br>
  -h, --help            show this help message and exit <br>
  --dir DIR             Directory with grib files. Default: current directory. <br>
  --out OUT             Output directory for csv files. Default: current directory. <br>
  --lat LAT             Latitude of point for which to get the data. Default: 54.6 <br>
  --long LONG           Longitude of point for which to get the data. Default: 18.8 <br>
  --shortName SHORTNAME
                        Short name of parameter to fetch. Default: 2t <br>
  --type TYPE           Type of level <br>
  --level LEVEL         Level <br>

# merge_csv.py
usage: merge_csv.py [-h] --dirs DIRS [DIRS ...] --names NAMES [NAMES ...] [--out OUT]

optional arguments: <br>
  -h, --help            show this help message and exit <br>
  --dirs DIRS [DIRS ...]
                        Input directories with csv files. Names of files should be in format YYYY-mm-dd-HHZ.csv <br>
  --names NAMES [NAMES ...]
                        Names of the parameters for each input csv <br>
  --out OUT             Output directory for csv files <br>



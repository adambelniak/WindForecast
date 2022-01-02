import os

TOO_MANY_REQUESTS = 'User has more than 10 open requests. Purge requests before trying again.'
REQUEST_TYPE_FIELD = "request_type"
REQUEST_STATUS_FIELD = "request_status"
REQUEST_ID_FIELD = "request_id"
NLAT_FIELD = "nlat"
SLAT_FIELD = "slat"
ELON_FIELD = "elon"
WLON_FIELD = "wlon"
LEVEL_FIELD = "level"
PARAM_FIELD = "param"
HOURS_TYPE_FIELD = "hours_type"

LAST_OFFSET_FOR_FORECAST = 120

RAW_CSV_FILENAME_WITH_REQUEST_REGEX = r'(gfs\.0p25\.\d{10}\.f\d{3}\.grib2\.[a-zA-Z]+)(\d+)(\.gp\.csv)'
RAW_NETCDF_FILENAME_WITH_REQUEST_REGEX = r'(gfs\.0p25\.(\d{10})\.f(\d{3})\.grib2\.[a-zA-Z]+)(\d+)(\.nc)'
RAW_NETCDF_FILENAME_REGEX = r'(gfs\.0p25\.(\d{10})\.f(\d{3})\.grib2(\.[a-zA-Z]+)?)(\.nc)'
RDA_CSV_FILENAME_FORMAT = 'gfs.0p25.{0}{1}{2}{3}.f{4}.grib2.{5}.gp.csv'
RDA_NETCDF_FILENAME_FORMAT = 'gfs.0p25.{0}{1}{2}{3}.f{4}.grib2.{5}.nc'
FINAL_CSV_FILENAME_FORMAT = '{0}-{1}-{2}-{3}Z.csv'
FINAL_NUMPY_FILENAME_FORMAT = '{0}-{1}-{2}-{3}-f{4}.npy'
WHOLE_YEAR_NUMPY_FILENAME_FORMAT = '{0}-{1}-f{2}.npy'

TAR_DOWNLOAD_PATH = os.path.join("D:\\", "WindForecast", "download", "tar")
CSV_DOWNLOAD_PATH = os.path.join("D:\\", "WindForecast", "download", "csv")
NETCDF_DOWNLOAD_PATH = os.path.join("D:\\", "WindForecast", "download", "netCDF")
# AI driven ~~wind~~ weather forecasting model. 

Trying to predict the most exact temperature, wind speed etc. for hours and days ahead
using LSTM, TCN, Transformer, NBEATSx etc. and some data analysis tools.

Seed for the analysis consists of:
 * GFS 0.25° archive forecasts from https://rda.ucar.edu
 * Synop reports from ground stations (https://danepubliczne.imgw.pl/)
 * Maximum Reflectivity images (CMAX) (https://danepubliczne.imgw.pl/datastore) 

The flow of getting GFS archive data is described in gfs-archive-0-25 module.

## Knowledge base
 * How to install pygrib: https://gist.github.com/emmanuelnk/406eee50c388f4f73dcdff521f2aa7b2?fbclid=IwAR3E3ErfD8WohQthZ2T4f95UkHb9-6v61w_l6Q5A5GC8u7nnQMeV90d8sNQ
  

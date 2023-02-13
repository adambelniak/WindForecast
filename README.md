# :sunny: AI driven weather forecasting model.
### :chart_with_upwards_trend: Weather forecasts as a point multivariate time series forecasting problem with Seq2Seq neural networks.

Trying to predict the most exact temperature, wind speed etc. for hours and days ahead
using LSTM, BiLSTM, TCN, Transformer, Spacetimeformer, NBEATSx. and some data analysis tools.

<b> Note: work is being performed in [MBelniak fork](https://github.com/MBelniak/WindForecast)<b>

## :card_index: Datasets

There are 3 different datasets used in this project. Based on experiment settings neural network uses datasets 1., 1. & 2. or 1. & 2. & 3.
1. Synop reports from ground stations [https://danepubliczne.imgw.pl/](https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/terminowe/synop/)
    - Multiple parameters are fetched and used, see [src/synop/consts.py](https://github.com/MBelniak/WindForecast/blob/master/src/synop/consts.py#L51)
    - wind velocity, direction and gusts are fetched from [https://danepubliczne.imgw.pl/datastore](https://danepubliczne.imgw.pl/datastore) for higher time and value resolution
2. GFS 0.25° archive forecasts from https://rda.ucar.edu
    - Multiple parameters are used, see [src/wind_forecast/config/train_parameters/CommonGFSConfig.json](https://github.com/MBelniak/WindForecast/blob/master/src/wind_forecast/config/train_parameters/CommonGFSConfig.json)
3. Maximum Reflectivity images (CMAX) [https://danepubliczne.imgw.pl/datastore](https://danepubliczne.imgw.pl/datastore)

The flow of getting GFS archive data is described in [gfs-archive-0-25](https://github.com/MBelniak/WindForecast/tree/master/src/gfs_archive_0_25) module.
Synop data is fetched in [src/synop/fetch_synop_data.py](https://github.com/MBelniak/WindForecast/blob/master/src/synop/fetch_synop_data.py).
CMAX data is fetched in [src/radar/fetch_radar_CMAX.py](https://github.com/MBelniak/WindForecast/blob/master/src/radar/fetch_radar_CMAX.py) and processed in [radar/preprocess_cmax.py](https://github.com/MBelniak/WindForecast/blob/master/src/radar/preprocess_cmax.py)

## :computer: Key technologies
  - [Pytorch](https://pytorch.org/) for creating models
  - [Pytorch Lightning](https://www.pytorchlightning.ai/) for training regime
  - [Weights & Biases](https://wandb.ai/) for logging and plotting results
  - [Hydra](https://hydra.cc/docs/intro/) for configuration
  - [Optuna](https://optuna.org/) for tuning
  - Numpy, Pandas, scikit-learn, matplotlib, seaborn as tooling 
  
## :bulb: Models
All models work in Seq2Seq fashion, with configurable time window and forecast horizon.
1. LSTM - encoder-decoder architecture with stacked LSTMs, as described in [Sequence to Sequence Learning with Neural Networks
](https://arxiv.org/abs/1409.3215)
2. BiLSTM - same as LSTM, but with bidirectional encoder
3. TCN - encoder-decoder architecture as described in [Temporal Convolutional Networks for the Advance Prediction of ENSO
](https://www.nature.com/articles/s41598-020-65070-5). There is also a model with just an encoder and a model with additional attention layers
4. Transformer - model based on [Attention is all you need](https://arxiv.org/abs/1706.03762)
5. Spacetimeformer - model based on [Long-Range Transformers for Dynamic Spatiotemporal Forecasting
](https://arxiv.org/abs/2109.12218)
6. NBeatsx - model based on [Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx
](https://arxiv.org/abs/2104.05522)

## :wrench: Configuration
There are several scopes in which an experiment can be configured. For tips on configuring run from command line see [scripts](https://github.com/MBelniak/WindForecast/tree/master/scripts). Also, see how to create predefined config via config files at [src/wind_forecast/config/experiment](https://github.com/MBelniak/WindForecast/tree/master/src/wind_forecast/config/experiment) or [src/wind_forecast/config/optim](https://github.com/MBelniak/WindForecast/tree/master/src/wind_forecast/config/optim).
1. config.experiment
    - training regime - nr of epochs, skip training, save checkpoint etc,
    - model specific config - models hyperparameters, dropout etc,
    - problem specific config - time window length, horizon length, target parameter, target location etc,
    - datasets config - val/test split, synop file, weather parameters to use, dates range
2. config.optim
    - lr, lr scheduler, optimizer, loss
3. config.lightning
    - deterministic training, gpus
4. config.tune - tune config; set of params to check

There are multiple configurations (yaml files) already prepared in [src/wind_forecast/config/experiment](src/wind_forecast/config/experiment), but they all use Sequence2SequenceWithCMAXDataModule, which requires CMAX files (reflectivity images). If you don't use CMAX files, better use Sequence2SequenceDataModule together with use_cmax_data: False and load_cmax_data: False. Sequence2SequenceWithCMAXDataModule is used in my experiments to have equal datasets across all experiments in my thesis.

## :running: Running
Obtaining datasets is described in [synop readme](src/synop/README.md), [GFS readme](src/gfs_archive_0_25/README.md) and [CMAX readme](src/radar/README.md).

Prepared synop data (csv file) should be placed in src/data/synop directory. There are already some files ready. Prepared GFS and CMAX datasets should be placed in a `pkl` directory placed in a directory pointed via GFS_DATASET_DIR and CMAX_DATASET_DIR environment variables.

<b>First, create conda environment<b>
```
conda env create -f environment.yml
```
<b>Then, install dependencies<b>
```
pip install -r requirements.txt
```

To run experiment, in `src` directory:
```
python -m wind_forecast.main experiment=<experiment_yml_file> [options...]
# e.g.
python -m wind_forecast.main experiment=transformer experiment.batch_size=32 lightning.gpus=0
```

### Run modes
RUN_MODE variable from `.env` file switches run mode. Do not specify in order to run a basic full training.
```
RUN_MODE=debug # Disables W&B logging and loads only a small part of datasets in order to start and perform the training process faster

RUN_MODE=tune # Performs tuning process. See [tune](https://github.com/MBelniak/WindForecast/tree/master/src/wind_forecast/config/tune) for examplary tune configs.

RUN_MODE=tune_debug # Joins the two above
```

### Weights & Biases
Add the following to `.env` to enable logging to W&B:
```
RESULTS_DIR=<relative to repo root, target dir for logs, checkpoints etc.>
WANDB_ENTITY=<your w&b username>
WANDB_PROJECT=<your w&b project name>
```

### Troubleshooting and tips
#### Faster dataloaders
```
 The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance
 ```
By default data is not loaded in parallel due to a problems on my Windows machine.
You can try speeding it up by setting `experiment.num_workers` to a number of cores on your machine  
or a smaller number if there are CUDA errors.


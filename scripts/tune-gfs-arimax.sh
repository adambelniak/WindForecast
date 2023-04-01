python -m wind_forecast.main experiment=arimax tune=arimax experiment.target_parameter=temperature lightning.gpus=0 \
experiment.batch_size=0 optim.optimizer._target_=None


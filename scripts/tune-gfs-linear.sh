python -m wind_forecast.main experiment=linear tune=linear experiment.target_parameter=temperature lightning.gpus=0 \
experiment.batch_size=0 optim.optimizer._target_=None


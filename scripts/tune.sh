python -m wind_forecast.main experiment=bi_lstm_s2s tune=bi_lstm_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=lstm_s2s tune=lstm_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=tcn_s2s tune=tcn_s2s experiment.epochs=10 optim=adam  experiment.target_parameter=temperature
python -m wind_forecast.main experiment=tcn_s2s_attention tune=tcn_s2s_attention experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=spacetimeformer tune=spacetimeformer experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=nbeatsx tune=nbeatsx experiment.epochs=10 optim=adam  experiment.target_parameter=temperature
python -m wind_forecast.main experiment=transformer tune=transformer experiment.epochs=80 optim=adam experiment.target_parameter=temperature

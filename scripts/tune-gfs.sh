python -m wind_forecast.main experiment=bi_lstm_s2s_gfs tune=bi_lstm_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs tune=bi_lstm_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=lstm_s2s_gfs tune=lstm_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs tune=lstm_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs tune=tcn_encoder_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=tcn_s2s_gfs tune=tcn_s2s experiment.epochs=10 optim=adam  experiment.target_parameter=temperature
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs tune=tcn_s2s experiment.epochs=10 optim=adam  experiment.target_parameter=temperature
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention tune=tcn_s2s_attention experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs tune=transformer_encoder_s2s experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=spacetimeformer_gfs tune=spacetimeformer experiment.epochs=10 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=nbeatsx_gfs tune=nbeatsx experiment.epochs=10 optim=adam  experiment.target_parameter=temperature
python -m wind_forecast.main experiment=transformer_gfs tune=transformer experiment.epochs=80 optim=adam experiment.target_parameter=temperature
python -m wind_forecast.main experiment=hybrid_transformer_gfs tune=transformer experiment.epochs=80 optim=adam experiment.target_parameter=temperature

#python -m wind_forecast.main experiment=arimax tune=arimax experiment.target_parameter=temperature lightning.gpus=0
#python -m wind_forecast.main experiment=sarimax tune=sarimax experiment.target_parameter=temperature lightning.gpus=0
#python -m wind_forecast.main experiment=sarimax tune=sarimax experiment.target_parameter=temperature experiment.sequence_length=48 lightning.gpus=0
#python -m wind_forecast.main experiment=sarimax tune=sarimax experiment.target_parameter=temperature experiment.sequence_length=72 lightning.gpus=0

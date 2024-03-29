python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs_cmax tune=bi_lstm_s2s_cmax experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs_cmax tune=lstm_s2s_cmax experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=tcn_encoder_s2s_cmax_gfs tune=tcn_encoder_s2s experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_cmax tune=tcn_s2s experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention_cmax tune=tcn_s2s_attention experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=transformer_encoder_s2s_cmax_gfs tune=transformer_encoder_s2s experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=spacetimeformer_gfs_cmax tune=spacetimeformer experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=nbeatsx_gfs_cmax tune=nbeatsx experiment.epochs=20 optim=adam experiment.target_parameter=temperature tune.trials=20
python -m wind_forecast.main experiment=hybrid_transformer_gfs_cmax tune=transformer experiment.epochs=80 optim=adam experiment.target_parameter=temperature tune.trials=20

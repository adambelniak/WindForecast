############# GFS #############
#NBEATSX
python -m wind_forecast.main experiment=nbeatsx_lstm_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=nbeatsx_lstm_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

python -m wind_forecast.main experiment=nbeatsx_lstm_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=nbeatsx_lstm_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

python -m wind_forecast.main experiment=nbeatsx_lstm_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=nbeatsx_lstm_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

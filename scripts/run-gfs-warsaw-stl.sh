############# GFS #############
#NBEATSX
python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

#LSTM
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

#BiLSTM
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

#TCN Encoder
python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

# TCN
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

# TCN Attention
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

  python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

  python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

# Transformer Encoder
#python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
#experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
#experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True
#
#python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
#experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
#experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True experiment.sequence_length=48
#
#python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
#experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
#experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
#experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True
#
#python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
#experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
#experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
#experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True experiment.sequence_length=48
#
#python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure \
#experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
#experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
#experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True
#
#python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure \
#experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
#experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
#experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True experiment.sequence_length=48
#

# Transformer
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 \
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 \
experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 \
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 \
experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 \
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 \
experiment.num_workers=16 experiment.stl_decompose=True

#Spacetimeformer
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.stl_decompose=True

## linear
#python -m wind_forecast.main experiment=linear experiment.target_parameter=temperature experiment.sequence_length=24 lightning.gpus=0
#python -m wind_forecast.main experiment=linear experiment.target_parameter=wind_velocity experiment.sequence_length=24 lightning.gpus=0
#python -m wind_forecast.main experiment=linear experiment.target_parameter=temperature=pressure experiment.sequence_length=24 lightning.gpus=0
#
#python -m wind_forecast.main experiment=linear experiment.target_parameter=temperature experiment.sequence_length=48 lightning.gpus=0
#python -m wind_forecast.main experiment=linear experiment.target_parameter=wind_velocity experiment.sequence_length=48 lightning.gpus=0
#python -m wind_forecast.main experiment=linear experiment.target_parameter=temperature=pressure experiment.sequence_length=48 lightning.gpus=0
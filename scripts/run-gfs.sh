############# GFS #############
#NBEATSX
python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.15 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=64 experiment.tcn_channels=[16,32] experiment.use_time2vec=False experiment.use_value2vec=True \
experiment.value2vec_embedding_factor=10 experiment.epochs=20

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.15 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=64 experiment.tcn_channels=[16,32] experiment.use_time2vec=False experiment.use_value2vec=True \
experiment.value2vec_embedding_factor=10 experiment.epochs=20 experiment.sequence_length=48

#python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.15 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[8,8] \
#experiment.nbeats_num_hidden=64 experiment.tcn_channels=[16,32] experiment.use_time2vec=False experiment.use_value2vec=True \
#experiment.value2vec_embedding_factor=10 experiment.epochs=20 experiment.sequence_length=48 experiment.future_sequence_length=48

#LSTM
python -m wind_forecast.main experiment=lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.5 optim.base_lr=0.0001 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=3 \
experiment.classification_head_dims=[128,128,64] experiment.epochs=20

python -m wind_forecast.main experiment=lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.5 optim.base_lr=0.0001 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=3 \
experiment.classification_head_dims=[128,128,64] experiment.epochs=20 experiment.sequence_length=48

#python -m wind_forecast.main experiment=lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.5 optim.base_lr=0.0001 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
#experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=3 \
#experiment.classification_head_dims=[128,128,64] experiment.epochs=20 experiment.sequence_length=48 experiment.future_sequence_length=48

#BiLSTM
python -m wind_forecast.main experiment=lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.05 optim.base_lr=0.0002 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.classification_head_dims=[64,32] experiment.epochs=20

python -m wind_forecast.main experiment=lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.05 optim.base_lr=0.0002 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.classification_head_dims=[64,32] experiment.epochs=20 \
experiment.sequence_length=48

#python -m wind_forecast.main experiment=lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.05 optim.base_lr=0.0002 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=8 \
#experiment.use_time2vec=False experiment.use_value2vec=False experiment.classification_head_dims=[64,32] experiment.epochs=20 \
#experiment.sequence_length=48 experiment.future_sequence_length=48

#TCN Encoder
python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.55 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[16,32] \
experiment.use_time2vec=True experiment.time2vec_embedding_factor=18 experiment.use_value2vec=False \
 experiment.classification_head_dims=[32] experiment.epochs=20

 python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.55 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[16,32] \
experiment.use_time2vec=True experiment.time2vec_embedding_factor=18 experiment.use_value2vec=False \
 experiment.classification_head_dims=[32] experiment.epochs=20 experiment.sequence_length=48

# python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.55 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[16,32] \
#experiment.use_time2vec=True experiment.time2vec_embedding_factor=18 experiment.use_value2vec=False \
# experiment.classification_head_dims=[32] experiment.epochs=20 experiment.sequence_length=48 experiment.future_sequence_length=48

# TCN
python -m wind_forecast.main experiment=tcn_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.55 optim.base_lr=0.00006 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64] \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=8 \
 experiment.classification_head_dims=[64,32] experiment.epochs=20

python -m wind_forecast.main experiment=tcn_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.55 optim.base_lr=0.00006 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64] \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=8 \
 experiment.classification_head_dims=[64,32] experiment.epochs=20 experiment.sequence_length=48

#python -m wind_forecast.main experiment=tcn_s2s_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.55 optim.base_lr=0.00006 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64] \
#experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=8 \
# experiment.classification_head_dims=[64,32] experiment.epochs=20 experiment.sequence_length=48 experiment.future_sequence_length=48

# TCN Attention
python -m wind_forecast.main experiment=tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.55 optim.base_lr=0.00005 experiment.tcn_kernel_size=5 experiment.tcn_channels=[32,64] \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=18 \
 experiment.classification_head_dims=[64,32] experiment.epochs=20

 python -m wind_forecast.main experiment=tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.55 optim.base_lr=0.00005 experiment.tcn_kernel_size=5 experiment.tcn_channels=[32,64] \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=18 \
 experiment.classification_head_dims=[64,32] experiment.epochs=20 experiment.sequence_length=48

# python -m wind_forecast.main experiment=tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.55 optim.base_lr=0.00005 experiment.tcn_kernel_size=5 experiment.tcn_channels=[32,64] \
#experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=18 \
# experiment.classification_head_dims=[64,32] experiment.epochs=20 experiment.sequence_length=48 experiment.future_sequence_length=48

# Transformer Encoder
python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.1 optim.base_lr=0.0002 experiment.transformer_ff_dim=128 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=True experiment.use_value2vec=False \
experiment.time2vec_embedding_factor=9 experiment.classification_head_dims=[32] experiment.epochs=20

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.1 optim.base_lr=0.0002 experiment.transformer_ff_dim=128 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=True experiment.use_value2vec=False \
experiment.time2vec_embedding_factor=9 experiment.classification_head_dims=[32] experiment.epochs=20 experiment.sequence_length=48

#python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.1 optim.base_lr=0.0002 experiment.transformer_ff_dim=128 \
#experiment.transformer_encoder_layers=6 experiment.use_time2vec=True experiment.use_value2vec=False \
#experiment.time2vec_embedding_factor=9 experiment.classification_head_dims=[32] experiment.epochs=20 experiment.sequence_length=48 experiment.future_sequence_length=48

# Transformer
python -m wind_forecast.main experiment=transformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.15 optim.base_lr=0.00025 experiment.teacher_forcing_epoch_num=20 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=16 \
 experiment.classification_head_dims=[64,32] experiment.epochs=30

python -m wind_forecast.main experiment=transformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.15 optim.base_lr=0.00025 experiment.teacher_forcing_epoch_num=20 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=16 \
 experiment.classification_head_dims=[64,32] experiment.epochs=30 experiment.sequence_length=48

#python -m wind_forecast.main experiment=transformer_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.15 optim.base_lr=0.00025 experiment.teacher_forcing_epoch_num=20 experiment.transformer_ff_dim=256 \
#experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=2 \
#experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=16 \
# experiment.classification_head_dims=[64,32] experiment.epochs=30 experiment.sequence_length=48 experiment.future_sequence_length=48

#Spacetimeformer
# TODO
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.07 optim.base_lr=0.0003 experiment.transformer_ff_dim=128 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.classification_head_dims=[64,32] experiment.epochs=20

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.07 optim.base_lr=0.0003 experiment.transformer_ff_dim=128 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.classification_head_dims=[64,32] \
 experiment.epochs=20 experiment.sequence_length=48

#python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature \
#experiment.dropout=0.07 optim.base_lr=0.0003 experiment.transformer_ff_dim=128 \
#experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
#experiment.use_time2vec=False experiment.use_value2vec=False experiment.classification_head_dims=[64,32] \
# experiment.epochs=20 experiment.sequence_length=48 experiment.future_sequence_length=48





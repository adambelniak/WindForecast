REM NBEATSX
REM python -m wind_forecast.main experiment=nbeatsx optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.4 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[4,4] ^
REM experiment.nbeats_num_hidden=64 experiment.tcn_channels=[32,64,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20

REM python -m wind_forecast.main experiment=nbeatsx optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.4 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[4,4] ^
REM experiment.nbeats_num_hidden=64 experiment.tcn_channels=[32,64,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20  experiment.sequence_length=48

REM python -m wind_forecast.main experiment=nbeatsx optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.4 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[4,4] ^
REM experiment.nbeats_num_hidden=64 experiment.tcn_channels=[32,64,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20

REM python -m wind_forecast.main experiment=nbeatsx optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.4 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[4,4] ^
REM experiment.nbeats_num_hidden=64 experiment.tcn_channels=[32,64,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20 experiment.sequence_length=48

REM python -m wind_forecast.main experiment=nbeatsx optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.4 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[4,4] ^
REM experiment.nbeats_num_hidden=64 experiment.tcn_channels=[32,64,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20

REM python -m wind_forecast.main experiment=nbeatsx optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.4 optim.base_lr=0.0002 experiment.nbeats_num_blocks=[4,4] experiment.nbeats_num_layers=[4,4] ^
REM experiment.nbeats_num_hidden=64 experiment.tcn_channels=[32,64,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20 experiment.sequence_length=48

REM LSTM
REM python -m wind_forecast.main experiment=lstm_s2s optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.06 optim.base_lr=0.00005 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=2 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,128,32] experiment.epochs=20

REM python -m wind_forecast.main experiment=lstm_s2s optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.06 optim.base_lr=0.00005 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=2 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.sequence_length=48

REM python -m wind_forecast.main experiment=lstm_s2s optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.06 optim.base_lr=0.00005 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=2 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,128,32] experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=lstm_s2s optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.06 optim.base_lr=0.00005 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=2 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.sequence_length=48
REM
REM python -m wind_forecast.main experiment=lstm_s2s optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.06 optim.base_lr=0.00005 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=2 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,128,32] experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=lstm_s2s optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.06 optim.base_lr=0.00005 experiment.lstm_hidden_state=1024 experiment.lstm_num_layers=2 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.sequence_length=48
REM
REM REM BiLSTM
REM python -m wind_forecast.main experiment=bi_lstm_s2s optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.1 optim.base_lr=0.0005 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[32] experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=bi_lstm_s2s optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.1 optim.base_lr=0.0005 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.sequence_length=48
REM
REM python -m wind_forecast.main experiment=bi_lstm_s2s optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.1 optim.base_lr=0.0005 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[32] experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=bi_lstm_s2s optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.1 optim.base_lr=0.0005 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.sequence_length=48
REM
REM python -m wind_forecast.main experiment=bi_lstm_s2s optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.1 optim.base_lr=0.0005 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[32] experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=bi_lstm_s2s optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.1 optim.base_lr=0.0005 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
REM experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.sequence_length=48
REM
REM REM TCN Encoder
REM REM python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
REM REM experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
REM REM experiment.epochs=20
REM REM
REM REM python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
REM REM experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
REM REM experiment.epochs=20 experiment.sequence_length=48
REM REM
REM REM python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
REM REM experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
REM REM experiment.epochs=20
REM REM
REM REM python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
REM REM experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
REM REM experiment.epochs=20 experiment.sequence_length=48
REM REM
REM REM python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
REM REM experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
REM REM experiment.epochs=20
REM REM
REM REM python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
REM REM experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
REM REM experiment.epochs=20 experiment.sequence_length=48
REM
REM REM  TCN
REM python -m wind_forecast.main experiment=tcn_s2s optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.6 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=tcn_s2s optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.6 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM experiment.epochs=20 experiment.sequence_length=48
REM
REM python -m wind_forecast.main experiment=tcn_s2s optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.6 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=tcn_s2s optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.6 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM experiment.epochs=20 experiment.sequence_length=48
REM
REM python -m wind_forecast.main experiment=tcn_s2s optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.6 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM experiment.epochs=20
REM
REM python -m wind_forecast.main experiment=tcn_s2s optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.6 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM experiment.epochs=20 experiment.sequence_length=48
REM
REM REM  TCN Attention
REM python -m wind_forecast.main experiment=tcn_s2s_attention optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.2 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM  experiment.epochs=20
REM
REM  python -m wind_forecast.main experiment=tcn_s2s_attention optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.2 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM  experiment.epochs=20 experiment.sequence_length=48

REM  python -m wind_forecast.main experiment=tcn_s2s_attention optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.2 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
REM  experiment.epochs=20

REM   python -m wind_forecast.main experiment=tcn_s2s_attention optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.2 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
REM experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
 REM experiment.epochs=20 experiment.sequence_length=48

  python -m wind_forecast.main experiment=tcn_s2s_attention optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
 experiment.epochs=20 

  python -m wind_forecast.main experiment=tcn_s2s_attention optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
 experiment.epochs=20 experiment.sequence_length=48


REM  Transformer Encoder
REM python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
REM experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,32] experiment.epochs=20 
REM 
REM python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
REM experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.sequence_length=48
REM 
REM python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
REM experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,32] experiment.epochs=20 
REM 
REM python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
REM experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
REM experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.sequence_length=48
REM 
REM python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
REM experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,32] experiment.epochs=20 
REM 
REM python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
REM experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
REM experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.sequence_length=48


REM  Transformer
python -m wind_forecast.main experiment=transformer optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.16 optim.base_lr=0.0001 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=40 ^


python -m wind_forecast.main experiment=transformer optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.16 optim.base_lr=0.0001 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=40 ^
experiment.sequence_length=48

python -m wind_forecast.main experiment=transformer optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.16 optim.base_lr=0.0001 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=40 ^


python -m wind_forecast.main experiment=transformer optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.16 optim.base_lr=0.0001 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=40 ^
experiment.sequence_length=48

python -m wind_forecast.main experiment=transformer optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.16 optim.base_lr=0.0001 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=40 ^


python -m wind_forecast.main experiment=transformer optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.16 optim.base_lr=0.0001 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=40 ^
experiment.sequence_length=48

REM Spacetimeformer
python -m wind_forecast.main experiment=spacetimeformer optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.13 optim.base_lr=0.006 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 

python -m wind_forecast.main experiment=spacetimeformer optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.13 optim.base_lr=0.006 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.sequence_length=48

python -m wind_forecast.main experiment=spacetimeformer optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.13 optim.base_lr=0.006 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 

python -m wind_forecast.main experiment=spacetimeformer optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.13 optim.base_lr=0.006 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.sequence_length=48

python -m wind_forecast.main experiment=spacetimeformer optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.13 optim.base_lr=0.006 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 

python -m wind_forecast.main experiment=spacetimeformer optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.13 optim.base_lr=0.006 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.sequence_length=48

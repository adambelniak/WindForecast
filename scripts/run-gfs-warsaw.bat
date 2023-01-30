REM NBEATSX
REM python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
REM experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-225mtjos:v0@model.ckpt
REM 
REM python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature ^
REM experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
REM experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
REM experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-2qaoaez6:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-336sa6h7:v0@model.ckpt

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-2koiwijw:v0@model.ckpt experiment.sequence_length=48

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3r67oahq:v0@model.ckpt

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] ^
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-14f6t2si:v0@model.ckpt experiment.sequence_length=48

REM LSTM
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-q7qnonsi:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-189of7qq:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-2hz7qdy8:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3fu9j692:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1s1y6upb:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.2 optim.base_lr=0.0002 experiment.lstm_hidden_state=512 experiment.lstm_num_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[32] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1ab4lf4l:v0@model.ckpt experiment.sequence_length=48
REM 
REM BiLSTM
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1rrza56t:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3u9rgo9g:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1z8via66:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1fcg4h78:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 ^
experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-30oaxz28:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.7 optim.base_lr=0.00045 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 ^
 experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[128,128,64] experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-urtw5s7g:v0@model.ckpt experiment.sequence_length=48
REM 
REM TCN Encoder
python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] ^
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

REM TCN
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-wno6d5th:v0@model.ckpt
REM
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3t18mr8a:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1m7c90xk:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-eqbnstwh:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-mlqv7x4r:v0@model.ckpt
REM
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-14mew6m9:v0@model.ckpt experiment.sequence_length=48
REM 
REM TCN Attention
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-2ip58ilb:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1r8q1utr:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-wsgkedlu:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-2z8z4nha:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-uus1c2ni:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-28uh6zif:v0@model.ckpt experiment.sequence_length=48
REM 
REM 
REM REM  Transformer Encoder
python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 ^
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False ^
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48


REM Transformer
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 ^
experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1xadcuh9:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 ^
experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3khr77ah:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 ^
experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-z364f6dz:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 ^
experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-2r0pdc9a:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 ^
experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-ghga597o:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.42 optim.base_lr=0.0007 experiment.teacher_forcing_epoch_num=40 experiment.transformer_ff_dim=128 ^
experiment.transformer_encoder_layers=6 experiment.transformer_decoder_layers=4 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] experiment.epochs=80 ^
experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-1s911z7i:v0@model.ckpt experiment.sequence_length=48
REM 
REM Spacetimeformer
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3i15g05v:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature ^
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3u289kbe:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3qqphx8u:v0@model.ckpt
REM 
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=wind_velocity ^
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3l407n63:v0@model.ckpt experiment.sequence_length=48
REM 
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-255ph8b2:v0@model.ckpt
REM
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=pressure ^
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 ^
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 ^
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] ^
experiment.epochs=20 experiment.resume_checkpoint=wandb://mbelniak/wind-forecast-openstack/model-3ospis7l:v0@model.ckpt experiment.sequence_length=48
REM 
REM GFS
python -m wind_forecast.main experiment=gfs experiment.target_parameter=temperature

python -m wind_forecast.main experiment=gfs experiment.target_parameter=wind_velocity

python -m wind_forecast.main experiment=gfs experiment.target_parameter=pressure
REM 
REM  linear
python -m wind_forecast.main experiment=linear experiment.target_parameter=temperature experiment.sequence_length=24 ^
experiment.skip_validation=True lightning.gpus=0 experiment.batch_size=0 optim.optimizer._target_=None experiment.linear_max_iter=10000 experiment.linear_L2_alpha=1

python -m wind_forecast.main experiment=linear experiment.target_parameter=wind_velocity experiment.sequence_length=24 ^
experiment.skip_validation=True lightning.gpus=0 experiment.batch_size=0 optim.optimizer._target_=None experiment.linear_max_iter=10000 experiment.linear_L2_alpha=1

python -m wind_forecast.main experiment=linear experiment.target_parameter=pressure experiment.sequence_length=24 ^
experiment.skip_validation=True lightning.gpus=0 experiment.batch_size=0 optim.optimizer._target_=None experiment.linear_max_iter=10000 experiment.linear_L2_alpha=1
REM 
REM arimax
python -m wind_forecast.main experiment=arimax experiment.target_parameter=temperature experiment.sequence_length=24 ^
experiment.skip_validation=True lightning.gpus=0 experiment.batch_size=0 optim.optimizer._target_=None experiment.arima_p=2 experiment.arima_d=1 experiment.arima_q=0
REM 
python -m wind_forecast.main experiment=arimax experiment.target_parameter=wind_velocity experiment.sequence_length=24 ^
experiment.skip_validation=True lightning.gpus=0 experiment.batch_size=0 optim.optimizer._target_=None experiment.arima_p=1 experiment.arima_d=1 experiment.arima_q=1
REM 
python -m wind_forecast.main experiment=arimax experiment.target_parameter=pressure experiment.sequence_length=24 ^
experiment.skip_validation=True lightning.gpus=0 experiment.batch_size=0 optim.optimizer._target_=None experiment.arima_p=3 experiment.arima_d=1 experiment.arima_q=0
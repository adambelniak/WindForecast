############# GFS #############
#NBEATSX
python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=nbeatsx_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.002 experiment.nbeats_num_blocks=[8,8] experiment.nbeats_num_layers=[8,8] \
experiment.nbeats_num_hidden=32 experiment.tcn_channels=[32,64] experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

#LSTM
python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.03 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=1 \
experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.03 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=1 \
experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.03 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=1 \
experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.03 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=1 \
experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.03 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=1 \
experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.03 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=2 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=1 \
experiment.regressor_head_dims=[64,128,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

#BiLSTM
python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=2 \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.2 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=2 \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=2 \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.2 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=2 \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=2 \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_bi_lstm_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.2 optim.base_lr=0.0003 experiment.lstm_hidden_state=256 experiment.lstm_num_layers=4 \
experiment.use_time2vec=False experiment.use_value2vec=True experiment.value2vec_embedding_factor=2 \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

#TCN Encoder
python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=tcn_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.3 optim.base_lr=0.0001 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[128,128,64] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

# TCN
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.05 optim.base_lr=0.0001 experiment.tcn_kernel_size=2 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

# TCN Attention
python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

 python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

  python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

  python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

  python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

  python -m wind_forecast.main experiment=hybrid_tcn_s2s_gfs_attention optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.6 optim.base_lr=0.0002 experiment.tcn_kernel_size=3 experiment.tcn_channels=[32,64,64] \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[32] \
 experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"


# Transformer Encoder
python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=transformer_encoder_s2s_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.7 optim.base_lr=0.008 experiment.transformer_ff_dim=512 \
experiment.transformer_encoder_layers=6 experiment.use_time2vec=False experiment.use_value2vec=False \
experiment.regressor_head_dims=[64,32] experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"


# Transformer
python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.03 optim.base_lr=0.000002 experiment.teacher_forcing_epoch_num=0 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=True experiment.use_value2vec=True experiment.value2vec_embedding_factor=15 \
experiment.time2vec_embedding_factor=20 experiment.regressor_head_dims=[32] experiment.epochs=40 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.03 optim.base_lr=0.000002 experiment.teacher_forcing_epoch_num=0 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=True experiment.use_value2vec=True experiment.value2vec_embedding_factor=15 \
experiment.time2vec_embedding_factor=20 experiment.regressor_head_dims=[32] experiment.epochs=40 \
experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.03 optim.base_lr=0.000002 experiment.teacher_forcing_epoch_num=0 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=True experiment.use_value2vec=True experiment.value2vec_embedding_factor=15 \
experiment.time2vec_embedding_factor=20 experiment.regressor_head_dims=[32] experiment.epochs=40 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.03 optim.base_lr=0.000002 experiment.teacher_forcing_epoch_num=0 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=True experiment.use_value2vec=True experiment.value2vec_embedding_factor=15 \
experiment.time2vec_embedding_factor=20 experiment.regressor_head_dims=[32] experiment.epochs=40 \
experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.03 optim.base_lr=0.000002 experiment.teacher_forcing_epoch_num=0 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=True experiment.use_value2vec=True experiment.value2vec_embedding_factor=15 \
experiment.time2vec_embedding_factor=20 experiment.regressor_head_dims=[32] experiment.epochs=40 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=hybrid_transformer_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.03 optim.base_lr=0.000002 experiment.teacher_forcing_epoch_num=0 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=2 \
experiment.use_time2vec=True experiment.use_value2vec=True experiment.value2vec_embedding_factor=15 \
experiment.time2vec_embedding_factor=20 experiment.regressor_head_dims=[32] experiment.epochs=40 \
experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

#Spacetimeformer
python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=temperature \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=wind_velocity \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=spacetimeformer_gfs optim=adam experiment.target_parameter=pressure \
experiment.dropout=0.5 optim.base_lr=0.001 experiment.transformer_ff_dim=256 \
experiment.transformer_encoder_layers=2 experiment.transformer_decoder_layers=8 \
experiment.use_time2vec=False experiment.use_value2vec=False experiment.regressor_head_dims=[64,128,32] \
experiment.epochs=20 experiment.num_workers=16 experiment.sequence_length=48 experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

#GFS
python -m wind_forecast.main experiment=gfs experiment.target_parameter=temperature experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=gfs experiment.target_parameter=wind_velocity experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

python -m wind_forecast.main experiment=gfs experiment.target_parameter=pressure experiment.target_coords=[54.829480163214775,18.33540561287926] experiment.synop_file="ROZEWIE_254180010_data.csv"

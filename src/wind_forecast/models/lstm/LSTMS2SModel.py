import math
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.value2vec.Value2Vec import Value2Vec
from wind_forecast.models.time2vec.Time2Vec import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class LSTMS2SModel(LightningModule):

    def __init__(self, config: Config):
        super(LSTMS2SModel, self).__init__()
        self.config = config
        self.lstm_hidden_state = config.experiment.lstm_hidden_state

        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        self.dropout = config.experiment.dropout
        self.features_length = len(config.experiment.synop_train_features) + len(config.experiment.synop_periodic_features)
        self.use_gfs = config.experiment.use_gfs_data
        self.gfs_on_head = config.experiment.gfs_on_head
        self.time2vec_embedding_factor = config.experiment.time2vec_embedding_factor
        self.value2vec_embedding_factor = config.experiment.value2vec_embedding_factor
        self.use_time2vec = config.experiment.use_time2vec and config.experiment.with_dates_inputs
        self.use_value2vec = config.experiment.use_value2vec and self.value2vec_embedding_factor > 0

        if not self.use_value2vec:
            self.value2vec_embedding_factor = 0

        if self.use_gfs:
            gfs_params = process_config(config.experiment.train_parameters_config_file).params
            gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                gfs_params_len += 1  # V and U will be expanded int velocity, sin and cos
            self.features_length += gfs_params_len

        if self.use_time2vec and self.time2vec_embedding_factor == 0:
            self.time2vec_embedding_factor = self.features_length

        self.dates_dim = self.config.experiment.dates_tensor_size * self.time2vec_embedding_factor if self.use_time2vec \
            else 2 * self.config.experiment.dates_tensor_size

        if self.use_time2vec:
            self.time_embed = TimeDistributed(Time2Vec(self.config.experiment.dates_tensor_size,
                                                       self.time2vec_embedding_factor), batch_first=True)

        if self.use_value2vec:
            self.value_embed = TimeDistributed(Value2Vec(self.features_length, self.value2vec_embedding_factor),
                                               batch_first=True)

        if config.experiment.with_dates_inputs:
            self.embed_dim = self.features_length * (self.value2vec_embedding_factor + 1) + self.dates_dim
        else:
            self.embed_dim = self.features_length * (self.value2vec_embedding_factor + 1)

        if self.embed_dim >= self.lstm_hidden_state:
            self.lstm_hidden_state = self.embed_dim + 1  # proj_size has to be smaller than hidden_size

        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.embed_dim)

        self.decoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.embed_dim)

        features = self.embed_dim
        if self.use_gfs and self.gfs_on_head:
            features += 1

        dense_layers = []
        for neurons in self.config.experiment.regressor_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))

        self.regressor_head = nn.Sequential(*dense_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         self.use_gfs, is_train)
        if self.use_gfs:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        _, state = self.encoder_lstm(input_elements)

        if epoch < self.teacher_forcing_epoch_num and is_train:
            # Teacher forcing
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.future_sequence_length)
                decoder_input = input_elements[:, -1:, :]
                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    next_pred, state = self.decoder_lstm(decoder_input, state)
                    pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                    decoder_input = next_pred[:, -1:, :]

                # then, do teacher forcing
                # SOS is appended for case when first_taught is 0
                decoder_input = torch.cat([input_elements[:, -1:, :], target_elements], 1)[:, first_taught:-1, ]
                next_pred, _ = self.decoder_lstm(decoder_input, state)
                output = torch.cat([pred, next_pred], -2) if pred is not None else next_pred

            else:
                # non-gradual, just basic teacher forcing
                decoder_input = torch.cat([input_elements[:, -1:, :], target_elements], 1)[:, :-1, ]
                output, _ = self.decoder_lstm(decoder_input, state)

        else:
            # inference - pass only predictions to decoder
            decoder_input = input_elements[:, -1:, :]
            pred = None
            for frame in range(self.future_sequence_length):
                next_pred, state = self.decoder_lstm(decoder_input, state)
                pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                decoder_input = next_pred[:, -1:, :]
            output = pred

        if self.use_gfs and self.gfs_on_head:
            return torch.squeeze(self.regressor_head(torch.cat([output, gfs_targets], -1)), -1)

        return torch.squeeze(self.regressor_head(output), -1)

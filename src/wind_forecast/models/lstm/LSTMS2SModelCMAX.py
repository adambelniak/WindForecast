import math
from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.lstm.LSTMS2SModel import LSTMS2SModel
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class LSTMS2SModelCMAX(LSTMS2SModel):

    def __init__(self, config: Config):
        super().__init__(config)
        self.config = config

        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        out_channels = config.experiment.cnn_filters[-1]
        self.conv = CMAXEncoder(config)
        for _ in config.experiment.cnn_filters:
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)

        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv, config)

        self.conv_time_distributed = TimeDistributed(self.conv, batch_first=True)

        self.embed_dim += conv_W * conv_H * out_channels

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

        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        if is_train:
            cmax_targets = batch[BatchKeys.CMAX_FUTURE.value].float()

        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))
        if is_train:
            self.conv_time_distributed.requires_grad_(False)
            cmax_targets_embeddings = self.conv_time_distributed(cmax_targets.unsqueeze(2))
            self.conv_time_distributed.requires_grad_(True)

        input_elements = torch.cat([input_elements, cmax_embeddings], -1)
        if is_train:
            target_elements = torch.cat([target_elements, cmax_targets_embeddings], -1)

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

import math
from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import get_pretrained_encoder, CMAXEncoder
from wind_forecast.models.lstm.HybridLSTMS2SModel import HybridLSTMS2SModel
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class HybridLSTMS2SCMAX(HybridLSTMS2SModel):

    def __init__(self, config: Config):
        super().__init__(config)
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
        self.decoder_embed_dim += conv_W * conv_H * out_channels
        self.decoder_output_dim = self.synop_features_length + conv_W * conv_H * out_channels

        if self.embed_dim >= self.lstm_hidden_state:
            self.lstm_hidden_state = self.embed_dim + 1  # proj_size has to be smaller than hidden_size

        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.decoder_output_dim)

        self.decoder_lstm = nn.LSTM(input_size=self.decoder_embed_dim, hidden_size=self.lstm_hidden_state,
                                    batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.decoder_output_dim)

        features = self.decoder_output_dim
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
        input_elements, all_synop_targets, all_gfs_targets, future_dates = self.get_embeddings(
            batch, self.config.experiment.with_dates_inputs,
            self.time_embed if self.use_time2vec else None,
            self.use_gfs, is_train)

        gfs_split_point = self.synop_features_length

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        _, state = self.encoder_lstm(input_elements)

        cmax_targets = batch[BatchKeys.CMAX_FUTURE.value].float()

        decoder_output = self.decoder_forward_with_cmax(epoch, is_train, state, input_elements, all_synop_targets,
                        all_gfs_targets, future_dates, cmax_targets)

        if self.use_gfs and self.gfs_on_head:
            return torch.squeeze(self.regressor_head(torch.cat([decoder_output, gfs_targets], -1)), -1)

        return torch.squeeze(self.regressor_head(decoder_output), -1)

    def decoder_forward_with_cmax(self, epoch, is_train, state, input_elements, all_synop_targets, all_gfs_targets,
                                  future_dates, cmax_targets):
        if epoch < self.teacher_forcing_epoch_num and is_train:
            self.conv.requires_grad_(False)
            cmax_targets_embeddings = self.conv_time_distributed(cmax_targets.unsqueeze(2))
            self.conv.requires_grad_(True)
            # Teacher forcing
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.future_sequence_length)
                decoder_input = torch.cat(
                    [
                        input_elements[:, -1:, :-(self.gfs_embed_dim + self.dates_dim)],
                        all_gfs_targets[:, :1, :],
                        future_dates[:, :1, :]
                    ], -1)

                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    next_pred, state = self.decoder_lstm(decoder_input, state)
                    pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                    if frame < first_taught - 1:
                        decoder_input = torch.cat(
                            [
                                next_pred[:, -1:, :],
                                all_gfs_targets[:, (frame + 1):(frame + 2), :],
                                future_dates[:, (frame + 1):(frame + 2), :]
                            ], -1)

                # then, do teacher forcing
                # SOS is appended for case when first_taught is 0
                first_decoder_input = torch.cat(
                    [
                        input_elements[:, -1:, :-(self.gfs_embed_dim + self.dates_dim)],
                        all_gfs_targets[:, :1, :],
                        future_dates[:, :1, :]
                    ], -1)
                next_decoder_inputs = torch.cat(
                    [
                        all_synop_targets,
                        cmax_targets_embeddings,
                        all_gfs_targets,
                        future_dates
                    ], -1)
                decoder_input = torch.cat([first_decoder_input, next_decoder_inputs], 1)[:, first_taught:-1, ]
                next_pred, _ = self.decoder_lstm(decoder_input, state)
                output = torch.cat([pred, next_pred], -2) if pred is not None else next_pred

            else:
                # non-gradual, just basic teacher forcing
                first_decoder_input = torch.cat(
                    [
                        input_elements[:, -1:, :-(self.gfs_embed_dim + self.dates_dim)],
                        all_gfs_targets[:, :1, :],
                        future_dates[:, :1, :]
                    ], -1)
                next_decoder_inputs = torch.cat(
                    [
                        all_synop_targets,
                        cmax_targets_embeddings,
                        all_gfs_targets,
                        future_dates
                    ], -1)
                decoder_input = torch.cat([first_decoder_input, next_decoder_inputs], 1)[:, :-1, ]
                output, _ = self.decoder_lstm(decoder_input, state)

        else:
            # inference - pass only predictions to decoder
            decoder_input = torch.cat(
                [
                    input_elements[:, -1:, :-(self.gfs_embed_dim + self.dates_dim)],
                    all_gfs_targets[:, :1, :],
                    future_dates[:, :1, :]
                ], -1)
            pred = None
            for frame in range(self.future_sequence_length):
                next_pred, state = self.decoder_lstm(decoder_input, state)
                pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                if frame < self.future_sequence_length - 1:
                    decoder_input = torch.cat(
                        [
                            next_pred[:, -1:, :],
                            all_gfs_targets[:, (frame + 1):(frame + 2), :],
                            future_dates[:, (frame + 1):(frame + 2), :]
                        ], -1)
            output = pred

        return output

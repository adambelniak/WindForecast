import math
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class LSTMS2SWithGFSModel(LightningModule):

    def __init__(self, config: Config):
        super(LSTMS2SWithGFSModel, self).__init__()
        self.config = config

        input_size = len(config.experiment.synop_train_features)

        if config.experiment.use_all_gfs_params:
            input_size += len(process_config(config.experiment.train_parameters_config_file))
        if config.experiment.with_dates_inputs:
            input_size += 2

        self.features_length = input_size
        self.embed_dim = self.features_length * (config.experiment.time2vec_embedding_size + 1)

        self.sequence_length = config.experiment.sequence_length
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        dropout = config.experiment.dropout
        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, batch_first=True,
                                    dropout=dropout, num_layers=config.experiment.lstm_num_layers)

        self.decoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, batch_first=True,
                                    dropout=dropout, num_layers=config.experiment.lstm_num_layers)

        self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(self.features_length,
                                                                    config.experiment.time2vec_embedding_size),
                                                           batch_first=True)

        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=self.embed_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.classification_head_time_distributed = TimeDistributed(self.dense, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        all_synop_targets = batch[BatchKeys.ALL_SYNOP_TARGETS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_EMBEDDING.value]

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                all_gfs_targets = batch[BatchKeys.ALL_GFS_TARGETS.value].float()
                x = [synop_inputs, gfs_inputs, dates_embedding[0], dates_embedding[1]]
                y = [all_synop_targets, all_gfs_targets, dates_embedding[2], dates_embedding[3]]
            else:
                x = [synop_inputs, dates_embedding[0], dates_embedding[1]]
                y = [all_synop_targets, dates_embedding[2], dates_embedding[3]]
        else:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                all_gfs_targets = batch[BatchKeys.ALL_GFS_TARGETS.value].float()
                x = [synop_inputs, gfs_inputs]
                y = [all_synop_targets, all_gfs_targets]
            else:
                x = [synop_inputs]
                y = [all_synop_targets]

        inputs = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)
        output, state = self.encoder_lstm(inputs)
        targets = torch.cat([*y, self.time_2_vec_time_distributed(torch.cat(y, -1))], -1)

        if epoch < self.teacher_forcing_epoch_num and stage in [None, 'fit']:
            # Teacher forcing
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.sequence_length)
                decoder_input = inputs[:, -1:, :]
                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    next_pred, state = self.decoder_lstm(decoder_input, state)
                    pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                    decoder_input = next_pred[:, -1:, :]

                # then, do teacher forcing
                # SOS is appended for case when first_taught is 0
                decoder_input = torch.cat([inputs[:, -1:, :], targets], 1)[:, first_taught:-1, ]
                next_pred, _ = self.decoder_lstm(decoder_input, state)
                output = torch.cat([pred, next_pred], -2) if pred is not None else next_pred

            else:
                # non-gradual, just basic teacher forcing
                decoder_input = torch.cat([inputs[:, -1:, :], targets], 1)[:, :-1, ]
                output, _ = self.decoder_lstm(decoder_input, state)

        else:
            # inference - pass only predictions to decoder
            decoder_input = inputs[:, -1:, :]
            pred = None
            for frame in range(synop_inputs.size(1)):
                next_pred, _ = self.decoder_lstm(decoder_input, state)
                pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                decoder_input = next_pred[:, -1:, :]
            output = pred

        return torch.squeeze(self.classification_head_time_distributed(output), -1)

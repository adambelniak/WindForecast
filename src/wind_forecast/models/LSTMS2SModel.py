import math
from typing import Dict

import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import Simple2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class LSTMS2SModel(LightningModule):

    def __init__(self, config: Config):
        super(LSTMS2SModel, self).__init__()
        self.config = config
        self.lstm_hidden_state = config.experiment.lstm_hidden_state

        input_size = len(config.experiment.synop_train_features) + len(config.experiment.periodic_features)
        self.use_gfs = config.experiment.use_gfs_data
        self.use_all_gfs = self.use_gfs and config.experiment.use_all_gfs_params

        if self.use_all_gfs:
            gfs_params = process_config(config.experiment.train_parameters_config_file)
            gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                gfs_params_len += 1  # V and U will be expanded int velocity, sin and cos
            input_size += gfs_params_len

        if config.experiment.with_dates_inputs:
            input_size += 6

        self.features_length = input_size
        self.embed_dim = self.features_length * (config.experiment.time2vec_embedding_size + 1)

        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        dropout = config.experiment.dropout
        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.embed_dim)

        self.decoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.embed_dim)

        self.simple_2_vec_time_distributed = TimeDistributed(Simple2Vec(self.features_length,
                                                                        config.experiment.time2vec_embedding_size),
                                                             batch_first=True)
        in_features = self.embed_dim
        if self.use_gfs:
            in_features += 1
        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )
        self.classification_head_time_distributed = TimeDistributed(self.dense, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        if is_train:
            all_synop_targets = batch[BatchKeys.SYNOP_FUTURE_X.value].float()

        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        if self.use_gfs:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()

        if self.config.experiment.with_dates_inputs:
            if self.use_all_gfs:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float()
                x = [synop_inputs, gfs_inputs, *dates_embedding[0]]
                if is_train:
                    y = [all_synop_targets, all_gfs_targets, *dates_embedding[1]]
            else:
                x = [synop_inputs, *dates_embedding[0]]
                if is_train:
                    y = [all_synop_targets, *dates_embedding[1]]
        else:
            if self.use_all_gfs:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float()
                x = [synop_inputs, gfs_inputs]
                if is_train:
                    y = [all_synop_targets, all_gfs_targets]
            else:
                x = [synop_inputs]
                if is_train:
                    y = [all_synop_targets]

        inputs = torch.cat([*x, self.simple_2_vec_time_distributed(torch.cat(x, -1))], -1)
        _, state = self.encoder_lstm(inputs)

        if is_train:
            targets = torch.cat([*y, self.simple_2_vec_time_distributed(torch.cat(y, -1))], -1)

        if epoch < self.teacher_forcing_epoch_num and is_train:
            # Teacher forcing
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.future_sequence_length)
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
            for frame in range(self.future_sequence_length):
                next_pred, state = self.decoder_lstm(decoder_input, state)
                pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                decoder_input = next_pred[:, -1:, :]
            output = pred
        if self.use_gfs:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets], -1)), -1)

        return torch.squeeze(self.classification_head_time_distributed(output), dim=-1)

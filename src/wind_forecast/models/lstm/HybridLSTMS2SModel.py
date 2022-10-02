import math
from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.lstm.LSTMS2SModel import LSTMS2SModel
from wind_forecast.models.value2vec.Value2Vec import Value2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class HybridLSTMS2SModel(LSTMS2SModel):

    def __init__(self, config: Config):
        super().__init__(config)
        assert self.use_gfs, "GFS needs to be used for hybrid model"
        self.synop_features_length = len(config.experiment.synop_train_features) + len(
            config.experiment.synop_periodic_features)

        self.decoder_output_dim = self.synop_features_length
        self.gfs_embed_dim = self.gfs_params_len

        if self.use_value2vec:
            self.decoder_output_dim += self.value2vec_embedding_factor * self.decoder_output_dim
            self.gfs_embed_dim += self.value2vec_embedding_factor * self.gfs_params_len
            self.value_embed_gfs = TimeDistributed(Value2Vec(self.gfs_params_len, self.value2vec_embedding_factor),
                                                   batch_first=True)
            self.value_embed_synop = TimeDistributed(Value2Vec(self.synop_features_length, self.value2vec_embedding_factor),
                                                   batch_first=True)


        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.decoder_output_dim)

        self.decoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state,
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
            batch, is_train)

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        _, state = self.encoder_lstm(input_elements)

        decoder_output = self.decoder_forward(epoch, is_train, state, input_elements, all_synop_targets,
                                              all_gfs_targets, future_dates)

        if self.use_gfs and self.gfs_on_head:
            return torch.squeeze(self.regressor_head(torch.cat([decoder_output, gfs_targets], -1)), -1)

        return torch.squeeze(self.regressor_head(decoder_output), -1)

    def get_embeddings(self, batch, with_future):
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        all_synop_targets = batch[BatchKeys.SYNOP_FUTURE_X.value].float() if with_future else None
        all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float() if self.use_gfs else None
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]
        future_dates = None

        if self.use_gfs:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            input_elements = torch.cat([synop_inputs, gfs_inputs], -1)
        else:
            input_elements = synop_inputs

        if self.use_value2vec:
            input_elements = torch.cat([input_elements, self.value_embed(input_elements)], -1)
            if with_future:
                all_synop_targets = torch.cat([all_synop_targets, self.value_embed_synop(all_synop_targets)], -1)
            all_gfs_targets = torch.cat([all_gfs_targets, self.value_embed_gfs(all_gfs_targets)], -1)

        if self.use_time2vec:
            input_elements = torch.cat([input_elements, self.time_embed(dates_tensors[0])], -1)
            future_dates = self.time_embed(dates_tensors[1])

        return input_elements, all_synop_targets, all_gfs_targets, future_dates

    def decoder_forward(self, epoch, is_train, state, input_elements, all_synop_targets, all_gfs_targets, future_dates):
        first_decoder_input = torch.cat(
            [
                input_elements[:, -1:, :-(self.gfs_embed_dim + self.dates_dim)],
                all_gfs_targets[:, :1, :],
                future_dates[:, :1, :]
            ], -1)
        if epoch < self.teacher_forcing_epoch_num and is_train:
            # Teacher forcing - not really evaluated, because accuracy not promising
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.future_sequence_length)
                decoder_input = first_decoder_input

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
                next_decoder_inputs = torch.cat(
                    [
                        all_synop_targets,
                        all_gfs_targets,
                        future_dates
                    ], -1)
                decoder_input = torch.cat([first_decoder_input, next_decoder_inputs], 1)[:, first_taught:-1, ]
                next_pred, _ = self.decoder_lstm(decoder_input, state)
                output = torch.cat([pred, next_pred], -2) if pred is not None else next_pred

            else:
                # non-gradual, just basic teacher forcing
                next_decoder_inputs = torch.cat(
                    [
                        all_synop_targets,
                        all_gfs_targets,
                        future_dates
                    ], -1)
                decoder_input = torch.cat([first_decoder_input, next_decoder_inputs], 1)[:, :-1, ]
                output, _ = self.decoder_lstm(decoder_input, state)

        else:
            # inference - pass only predictions to decoder
            decoder_input = first_decoder_input
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

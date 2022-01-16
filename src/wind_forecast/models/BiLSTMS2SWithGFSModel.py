from typing import Dict

from torch import nn
import torch
import math
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.LSTMS2SWithGFSModel import LSTMS2SWithGFSModel
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class BiLSTMS2SWithGFSModel(LSTMS2SWithGFSModel):

    def __init__(self, config: Config):
        super(BiLSTMS2SWithGFSModel, self).__init__(config)
        dropout = config.experiment.dropout
        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, batch_first=True,
                                    dropout=dropout, num_layers=config.experiment.lstm_num_layers, bidirectional=True)

        self.decoder_first_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=2 * self.embed_dim, batch_first=True,
                                          dropout=dropout)

        self.decoder_last_lstm = nn.LSTM(input_size=2 * self.embed_dim, hidden_size=self.embed_dim, batch_first=True,
                                         dropout=dropout)
        self.decoder_lstm = None
        if config.experiment.lstm_num_layers > 1:
            self.decoder_lstm = nn.LSTM(input_size=2 * self.embed_dim, hidden_size=2 * self.embed_dim, batch_first=True,
                                        dropout=dropout, num_layers=config.experiment.lstm_num_layers - 1)

        self.state_dense = nn.Linear(in_features=2 * self.embed_dim, out_features=2 * self.embed_dim)
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
            BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                all_gfs_targets = batch[BatchKeys.ALL_GFS_TARGETS.value].float()
                x = [synop_inputs, gfs_inputs, *dates_embedding[0], *dates_embedding[1]]
                y = [all_synop_targets, all_gfs_targets, *dates_embedding[2], *dates_embedding[3]]
            else:
                x = [synop_inputs, *dates_embedding[0], *dates_embedding[1]]
                y = [all_synop_targets, *dates_embedding[2], *dates_embedding[3]]
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
        state = (torch.cat([state[0][0:self.config.experiment.lstm_num_layers, :, :],
                            state[0][self.config.experiment.lstm_num_layers:, :, :]], -1),
                 torch.cat([state[1][0:self.config.experiment.lstm_num_layers, :, :],
                            state[1][self.config.experiment.lstm_num_layers:, :, :]], -1))
        state = (self.state_dense(state[0]), self.state_dense(state[1]))
        targets = torch.cat([*y, self.time_2_vec_time_distributed(torch.cat(y, -1))], -1)

        if epoch < self.teacher_forcing_epoch_num and stage in [None, 'fit']:
            # Teacher forcing
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.sequence_length)
                decoder_input = inputs[:, -1:, :]
                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    first_pred, first_state = self.decoder_first_lstm(decoder_input,
                                                                      (state[0][0:1, :, :], state[1][0:1, :, :]))
                    if self.decoder_lstm is not None:
                        next_pred, state = self.decoder_lstm(first_pred, (state[0][1:, :, :], state[1][1:, :, :]))
                        last_pred, last_state = self.decoder_last_lstm(next_pred)
                    else:
                        last_pred, last_state = self.decoder_last_lstm(first_pred)

                    pred = torch.cat([pred, last_pred[:, -1:, :]], -2) if pred is not None else last_pred[:, -1:, :]
                    decoder_input = last_pred[:, -1:, :]
                    state = (torch.cat([first_state[0], state[0]], dim=0), torch.cat([first_state[1], state[1]], dim=0))

                # then, do teacher forcing
                # SOS is appended for case when first_taught is 0
                decoder_input = torch.cat([inputs[:, -1:, :], targets], 1)[:, first_taught:-1, ]
                first_pred, first_state = self.decoder_first_lstm(decoder_input,
                                                                  (state[0][0:1, :, :], state[1][0:1, :, :]))
                if self.decoder_lstm is not None:
                    next_pred, state = self.decoder_lstm(first_pred, (state[0][1:, :, :], state[1][1:, :, :]))
                    last_pred, last_state = self.decoder_last_lstm(next_pred)
                else:
                    last_pred, last_state = self.decoder_last_lstm(first_pred)
                output = torch.cat([pred, last_pred], -2) if pred is not None else last_pred

            else:
                # non-gradual, just basic teacher forcing
                decoder_input = torch.cat([inputs[:, -1:, :], targets], 1)[:, :-1, ]
                first_pred, first_state = self.decoder_first_lstm(decoder_input,
                                                                  (state[0][0:1, :, :], state[1][0:1, :, :]))
                if self.decoder_lstm is not None:
                    next_pred, state = self.decoder_lstm(first_pred, (state[0][1:, :, :], state[1][1:, :, :]))
                    output, _ = self.decoder_last_lstm(next_pred)
                else:
                    output, _ = self.decoder_last_lstm(first_pred)

        else:
            # inference - pass only predictions to decoder
            decoder_input = inputs[:, -1:, :]
            pred = None
            for frame in range(synop_inputs.size(1)):
                first_pred, first_state = self.decoder_first_lstm(decoder_input,
                                                                  (state[0][0:1, :, :], state[1][0:1, :, :]))
                if self.decoder_lstm is not None:
                    next_pred, state = self.decoder_lstm(first_pred, (state[0][1:, :, :], state[1][1:, :, :]))
                    last_pred, last_state = self.decoder_last_lstm(next_pred)
                else:
                    last_pred, last_state = self.decoder_last_lstm(first_pred)
                pred = torch.cat([pred, last_pred[:, -1:, :]], -2) if pred is not None else last_pred[:, -1:, :]
                decoder_input = last_pred[:, -1:, :]
                state = (torch.cat([first_state[0], state[0]], dim=0), torch.cat([first_state[1], state[1]], dim=0))
            output = pred

        return torch.squeeze(self.classification_head_time_distributed(output), -1)

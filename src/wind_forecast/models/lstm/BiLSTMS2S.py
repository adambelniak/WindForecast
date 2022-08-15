from typing import Dict

from torch import nn
import torch
import math
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.lstm.LSTMS2SModel import LSTMS2SModel


class BiLSTMS2S(LSTMS2SModel):

    def __init__(self, config: Config):
        super(BiLSTMS2S, self).__init__(config)
        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.embed_dim, bidirectional=True)

        self.decoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=2 * self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.embed_dim)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         self.use_gfs, is_train)
        if self.use_gfs:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()

        output, state = self.encoder_lstm(input_elements)
        # state is of shape ((2 * num_layers, batch, H_out), (2 * num_layers, batch, H_cell)
        # concatenate forward and backward states on the last axis for cell_state,
        # use only forward out state
        state = (state[0][0:self.config.experiment.lstm_num_layers, :, :],
                 torch.cat([state[1][0:self.config.experiment.lstm_num_layers, :, :],
                            state[1][self.config.experiment.lstm_num_layers:, :, :]], -1))

        if epoch < self.teacher_forcing_epoch_num and stage in [None, 'fit']:
            # Teacher forcing
            if self.gradual_teacher_forcing:
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.sequence_length)
                decoder_input = input_elements[:, -1:, :]
                pred = None
                for frame in range(first_taught):  # do normal prediction for the beginning frames
                    next_pred, state = self.decoder_lstm(decoder_input, state)
                    pred = torch.cat([pred, next_pred[:, -1:, :]], -2) if pred is not None else next_pred[:, -1:, :]
                    decoder_input = next_pred[:, -1:, :]

                # then, do teacher forcing
                # SOS is appended for case when first_taught is 0
                decoder_input = torch.cat([input_elements[:, -1:, :], target_elements], 1)[:, first_taught:-1, ]
                next_pred, state = self.decoder_lstm(decoder_input, state)

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
            return torch.squeeze(self.classification_head(torch.cat([output, gfs_targets], -1)), -1)

        return torch.squeeze(self.classification_head(output), -1)

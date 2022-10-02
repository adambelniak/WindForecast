from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.lstm.HybridLSTMS2SCMAX import HybridLSTMS2SCMAX


class HybridBiLSTMS2SCMAX(HybridLSTMS2SCMAX):

    def __init__(self, config: Config):
        super().__init__(config)

        self.encoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.decoder_output_dim, bidirectional=True)

        self.decoder_lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=2 * self.lstm_hidden_state, batch_first=True,
                                    dropout=self.dropout, num_layers=config.experiment.lstm_num_layers,
                                    proj_size=self.decoder_output_dim)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, all_synop_targets, all_gfs_targets, cmax_future, future_dates = self.get_embeddings_cmax(
            batch, self.time_embed, is_train)

        output, state = self.encoder_lstm(input_elements)
        # state is of shape ((2 * num_layers, batch, H_out), (2 * num_layers, batch, H_cell)
        # concatenate forward and backward states on the last axis for cell_state,
        # use only forward out state
        state = (state[0][0:self.config.experiment.lstm_num_layers, :, :],
                 torch.cat([state[1][0:self.config.experiment.lstm_num_layers, :, :],
                            state[1][self.config.experiment.lstm_num_layers:, :, :]], -1))

        decoder_output = self.decoder_forward_with_cmax(epoch, is_train, state, input_elements, all_synop_targets,
                                                        all_gfs_targets, future_dates, cmax_future)

        if self.use_gfs and self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return torch.squeeze(self.regressor_head(torch.cat([decoder_output, gfs_targets], -1)), -1)

        return torch.squeeze(self.regressor_head(decoder_output), -1)

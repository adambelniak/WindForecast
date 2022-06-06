from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlockWithAttention
from wind_forecast.models.TCNS2SWithDecoder import TemporalConvNetS2SWithDecoder
from wind_forecast.util.config import process_config


class TCNS2SWithDecoderModelWithAttention(TemporalConvNetS2SWithDecoder):
    def __init__(self, config: Config):
        super(TCNS2SWithDecoderModelWithAttention, self).__init__(config)

        in_channels = len(config.experiment.synop_train_features) + len(config.experiment.periodic_features)
        if self.use_gfs_on_input:
            gfs_params = process_config(config.experiment.train_parameters_config_file)
            gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                gfs_params_len += 1  # V and U will be expanded int velocity, sin and cos
            in_channels += gfs_params_len

        if config.experiment.with_dates_inputs:
            in_channels += 6

        tcn_layers = []

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            out_channels = self.tcn_channels[i]
            tcn_layers += [TemporalBlockWithAttention(config.experiment.transformer_attention_heads,
                                                      in_channels, out_channels, self.kernel_size,
                                                      dilation=dilation_size,
                                                      padding=(self.kernel_size - 1) * dilation_size,
                                                      dropout=self.dropout)]
            in_channels = self.tcn_channels[i]

        self.encoder = nn.Sequential(*tcn_layers)

        tcn_layers = []

        for i in range(self.num_levels-1):
            dilation_size = 2 ** (self.num_levels - i)
            out_channels = self.tcn_channels[-(i+2)]
            tcn_layers += [TemporalBlockWithAttention(config.experiment.transformer_attention_heads,
                                                      in_channels, out_channels, self.kernel_size,
                                                      dilation=dilation_size,
                                                      padding=(self.kernel_size - 1) * dilation_size,
                                                      dropout=self.dropout)]
            in_channels = out_channels

        self.decoder = nn.Sequential(*tcn_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        gfs_targets = None if not self.use_gfs else batch[BatchKeys.GFS_FUTURE_Y.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.with_dates_inputs:
            if self.use_gfs_on_input:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                x = [synop_inputs, gfs_inputs, *dates_embedding[0]]
            else:
                x = [synop_inputs, *dates_embedding[0]]
        else:
            if self.use_gfs_on_input:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                x = [synop_inputs, gfs_inputs]
            else:
                x = [synop_inputs]

        x = torch.cat(x, -1).permute(0, 2, 1)
        x = self.encoder(x)
        mem = self.hidden_space_lin(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.decoder(mem)

        if self.config.experiment.with_dates_inputs:
            if self.use_gfs:
                return self.linear_time_distributed(torch.cat(
                    [y.permute(0, 2, 1)[:, -self.future_sequence_length:, :], gfs_targets, *dates_embedding[1]],
                    -1)).squeeze(-1)
            return self.linear_time_distributed(
                torch.cat([y.permute(0, 2, 1)[:, -self.future_sequence_length:, :], *dates_embedding[1]], -1)).squeeze(
                -1)
        if self.use_gfs:
            return self.linear_time_distributed(
                torch.cat([y.permute(0, 2, 1)[:, -self.future_sequence_length:, :], gfs_targets], -1)).squeeze(-1)
        return self.linear_time_distributed(y.permute(0, 2, 1)[:, -self.future_sequence_length:, :]).squeeze(-1)

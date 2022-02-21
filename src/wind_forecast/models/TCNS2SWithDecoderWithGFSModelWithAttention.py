from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlockWithAttention
from wind_forecast.models.TCNS2SWithDecoderWithGFSModel import TemporalConvNetS2SWithDencoderWithGFS
from wind_forecast.util.config import process_config


class TCNS2SWithDecoderWithGFSModelWithAttention(TemporalConvNetS2SWithDencoderWithGFS):
    def __init__(self, config: Config):
        super(TCNS2SWithDecoderWithGFSModelWithAttention, self).__init__(config)

        in_channels = len(config.experiment.synop_train_features)
        if config.experiment.use_all_gfs_params:
            in_channels += len(process_config(config.experiment.train_parameters_config_file))

        if config.experiment.with_dates_inputs:
            in_channels += 4
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

        if config.experiment.with_dates_inputs:
            # gfs_targets + dates
            in_channels += 5
        else:
            # gfs_targets
            in_channels += 1

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            out_channels = self.tcn_channels[i]
            tcn_layers += [TemporalBlockWithAttention(config.experiment.transformer_attention_heads,
                                                      in_channels, out_channels, self.kernel_size,
                                                      dilation=dilation_size,
                                                      padding=(self.kernel_size - 1) * dilation_size,
                                                      dropout=self.dropout)]
            in_channels = self.tcn_channels[i]

        self.decoder = nn.Sequential(*tcn_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        gfs_targets = batch[BatchKeys.GFS_TARGETS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs, *dates_embedding[0]]
            else:
                x = [synop_inputs, *dates_embedding[0]]
        else:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs]
            else:
                x = [synop_inputs]

        x = torch.cat(x, -1).permute(0, 2, 1)
        mem = self.encoder(x)
        mem = mem[:, -self.future_sequence_length:, :]
        if self.config.experiment.with_dates_inputs:
            decoder_input = torch.cat([mem, gfs_targets.permute(0, 2, 1), *dates_embedding[1].permute(0, 2, 1)], -2)
        else:
            decoder_input = torch.cat([mem, gfs_targets.permute(0, 2, 1)], -2)

        y = self.decoder(decoder_input)
        return self.linear_time_distributed(y.permute(0, 2, 1)).squeeze(-1)

from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TemporalConvNetS2SWithDecoder(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNetS2SWithDecoder, self).__init__()
        self.config = config
        self.use_gfs = config.experiment.use_gfs_data
        self.use_gfs_on_input = self.use_gfs and config.experiment.use_all_gfs_params
        self.future_sequence_length = config.experiment.future_sequence_length
        self.dropout = config.experiment.dropout
        self.tcn_channels = config.experiment.tcn_channels
        self.num_levels = len(self.tcn_channels)
        self.kernel_size = config.experiment.tcn_kernel_size

        in_channels = len(config.experiment.synop_train_features) + len(config.experiment.periodic_features)
        if self.use_gfs_on_input:
            in_channels += len(process_config(config.experiment.train_parameters_config_file))

        if config.experiment.with_dates_inputs:
            in_channels += 6

        tcn_layers = []

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            out_channels = self.tcn_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, dilation=dilation_size,
                                         padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
            in_channels = out_channels

        self.encoder = nn.Sequential(*tcn_layers)
        self.hidden_space_lin = TimeDistributed(nn.Linear(in_features=in_channels, out_features=in_channels), batch_first=True)

        tcn_layers = []

        for i in range(self.num_levels-1):
            dilation_size = 2 ** (self.num_levels - i)
            out_channels = self.tcn_channels[-(i+2)]
            tcn_layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, dilation=dilation_size,
                                         padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
            in_channels = out_channels

        self.decoder = nn.Sequential(*tcn_layers)
        in_features = self.tcn_channels[0]

        if config.experiment.with_dates_inputs:
            in_features += 6
        if self.use_gfs:
            in_features += 1

        linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

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

        x = self.encoder(torch.cat(x, -1).permute(0, 2, 1))
        mem = self.hidden_space_lin(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.decoder(mem)

        if self.config.experiment.with_dates_inputs:
            if self.use_gfs:
                return self.linear_time_distributed(torch.cat([y.permute(0, 2, 1)[:, -self.future_sequence_length:, :], gfs_targets, *dates_embedding[1]], -1)).squeeze(-1)
            return self.linear_time_distributed(torch.cat([y.permute(0, 2, 1)[:, -self.future_sequence_length:, :], *dates_embedding[1]], -1)).squeeze(-1)
        if self.use_gfs:
            return self.linear_time_distributed(torch.cat([y.permute(0, 2, 1)[:, -self.future_sequence_length:, :], gfs_targets], -1)).squeeze(-1)
        return self.linear_time_distributed(y.permute(0, 2, 1)[:, -self.future_sequence_length:, :]).squeeze(-1)

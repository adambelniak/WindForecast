from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TemporalConvNetS2SWithDencoderWithGFS(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNetS2SWithDencoderWithGFS, self).__init__()
        self.config = config
        dropout = config.experiment.dropout
        tcn_layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        in_channels = len(config.experiment.synop_train_features)
        if config.experiment.use_all_gfs_params:
            in_channels += len(process_config(config.experiment.train_parameters_config_file))

        if config.experiment.with_dates_inputs:
            in_channels += 2

        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            in_channels = num_channels[i]

        self.encoder = nn.Sequential(*tcn_layers)

        tcn_layers = []

        if config.experiment.with_dates_inputs:
            # gfs_targets + dates
            in_channels += 3
        else:
            # gfs_targets
            in_channels += 1

        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
            in_channels = num_channels[i]

        self.decoder = nn.Sequential(*tcn_layers)
        in_features = in_channels

        linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        gfs_targets = batch[BatchKeys.GFS_TARGETS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_EMBEDDING.value]

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs, dates_embedding[0], dates_embedding[1]]
            else:
                x = [synop_inputs, dates_embedding[0], dates_embedding[1]]
        else:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs]
            else:
                x = [synop_inputs]

        x = self.encoder(torch.cat(x, -1).permute(0, 2, 1))
        if self.config.experiment.with_dates_inputs:
            y = self.decoder(torch.cat([x, gfs_targets.permute(0, 2, 1), dates_embedding[2].permute(0, 2, 1), dates_embedding[3].permute(0, 2, 1)], -2))
        else:
            y = self.decoder(torch.cat([x, gfs_targets.permute(0, 2, 1)], -2))

        return self.linear_time_distributed(y.permute(0, 2, 1)).squeeze(-1)

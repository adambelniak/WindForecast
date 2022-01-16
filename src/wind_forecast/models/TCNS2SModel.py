from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TemporalConvNetS2S(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNetS2S, self).__init__()
        self.config = config
        tcn_layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        in_channels = len(config.experiment.synop_train_features)
        if config.experiment.with_dates_inputs:
            in_channels += 4
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=config.experiment.dropout)]
            in_channels = out_channels

        if self.config.experiment.with_dates_inputs:
            features = num_channels[-1] + 4
        else:
            features = num_channels[-1]

        linear = nn.Sequential(
            nn.Linear(in_features=features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

        self.tcn = nn.Sequential(*tcn_layers)
        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.with_dates_inputs:
            x = [synop_inputs, *dates_embedding[0], *dates_embedding[1]]
        else:
            x = [synop_inputs]
        x = self.tcn(torch.cat(x, dim=-1).permute(0, 2, 1))

        if self.config.experiment.with_dates_inputs:
            return self.linear_time_distributed(torch.cat([x.permute(0, 2, 1), *dates_embedding[2], *dates_embedding[3]], -1)).squeeze(-1)
        else:
            return self.linear_time_distributed(x.permute(0, 2, 1)).squeeze(-1)

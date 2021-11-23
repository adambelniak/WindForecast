from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TCNS2SCMAX(LightningModule):
    def __init__(self, config: Config):
        super(TCNS2SCMAX, self).__init__()
        self.cnn = TimeDistributed(self.create_cnn_layers(config), batch_first=True)
        self.cnn_lin_tcn = TimeDistributed(nn.Linear(in_features=config.experiment.cnn_lin_tcn_in_features,
                                                     out_features=config.experiment.tcn_channels[0] - len(
                                                         config.experiment.synop_train_features)),
                                           batch_first=True)
        self.tcn = self.create_tcn_layers(config)

        if self.config.experiment.with_dates_inputs:
            features = config.experiment.tcn_channels + 2
        else:
            features = config.experiment.tcn_channels

        linear = nn.Sequential(
            nn.Linear(in_features=features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

        self.linear = TimeDistributed(linear, batch_first=True)

    def create_cnn_layers(self, config: Config):
        cnn_channels = len(process_config(config.experiment.train_parameters_config_file))
        cnn_layers = []

        for index, filters in enumerate(config.experiment.cnn_filters):
            cnn_layers.append(
                nn.Conv2d(in_channels=cnn_channels, out_channels=filters, kernel_size=(3, 3), padding=(1, 1),
                          stride=(2, 2)))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.BatchNorm2d(num_features=filters))
            if index != len(config.experiment.cnn_filters) - 1:
                cnn_layers.append(nn.Dropout(config.experiment.dropout))
            cnn_channels = filters

        cnn_layers.append(nn.Flatten())
        return nn.Sequential(*cnn_layers)

    def create_tcn_layers(self, config: Config):
        tcn_layers = []
        tcn_channels = config.experiment.tcn_channels
        kernel_size = config.experiment.tcn_kernel_size
        for i in range(len(tcn_channels) - 1):
            dilation_size = 2 ** i
            in_channels = tcn_channels[i]
            out_channels = tcn_channels[i + 1]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size)]

        return nn.Sequential(*tcn_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_INPUTS.value].float()

        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_EMBEDDING.value]

        if self.config.experiment.with_dates_inputs:
            x = [synop_inputs, dates_embedding[0], dates_embedding[1]]
        else:
            x = [synop_inputs]

        cmax_embedding = self.cnn(cmax_inputs.unsqueeze(2))
        cmax_embedding = self.cnn_lin_tcn(cmax_embedding)
        x = torch.cat([*x, cmax_embedding], dim=-1)
        x = self.tcn(x.permute(0, 2, 1))

        if self.config.experiment.with_dates_inputs:
            return self.linear_time_distributed(torch.cat([x.permute(0, 2, 1), dates_embedding[2], dates_embedding[3]], -1)).squeeze(-1)
        else:
            return self.linear_time_distributed(x.permute(0, 2, 1)).squeeze(-1)

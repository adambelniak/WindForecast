import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn.utils import weight_norm
from wind_forecast.config.register import Config
from wind_forecast.models.TCNModel import TemporalBlock


class TemporalConvNet(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = len(config.experiment.synop_train_features) if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size)]

        self.convolutional = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=num_channels[-1] * config.experiment.sequence_length + 1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, synop_input, gfs_input) -> torch.Tensor:
        x = synop_input.permute(0, 2, 1)
        x = self.convolutional(x)
        x = self.flatten(x)
        x = self.feed_forward(torch.cat((x, gfs_input.unsqueeze(-1)), dim=1))
        return x.squeeze()

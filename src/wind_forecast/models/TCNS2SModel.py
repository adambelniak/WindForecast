import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from wind_forecast.config.register import Config
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TemporalConvNetS2S(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNetS2S, self).__init__()
        tcn_layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = len(config.experiment.synop_train_features) if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size)]

        linear = nn.Sequential(
            nn.Linear(in_features=num_channels[-1], out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.tcn = nn.Sequential(*tcn_layers)
        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int, stage=None) -> torch.Tensor:
        x = self.tcn(inputs.permute(0, 2, 1)).squeeze()
        return self.linear_time_distributed(x.permute(0, 2, 1)).squeeze()

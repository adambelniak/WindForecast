import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from wind_forecast.config.register import Config
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TemporalConvNetS2SWithGFS(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNetS2SWithGFS, self).__init__()
        tcn_layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        in_channels = len(config.experiment.synop_train_features)
        if config.experiment.use_all_gfs_as_input:
            in_channels += len(process_config(config.experiment.train_parameters_config_file))

        if config.experiment.with_dates_inputs:
            in_channels += 2

        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size)]
            in_channels = num_channels[i]

        if config.experiment.with_dates_inputs:
            in_features = num_channels[-1] + 3
        else:
            in_features = num_channels[-1] + 1

        linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.tcn = nn.Sequential(*tcn_layers)
        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def forward(self, inputs, gfs_inputs, gfs_targets, targets: torch.Tensor, epoch: int, stage=None,
                dates_embeddings: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) = None) -> torch.Tensor:
        if gfs_inputs is None:
            if dates_embeddings is None:
                x = [inputs]
            else:
                x = [inputs, dates_embeddings[0], dates_embeddings[1]]
            x = self.tcn(torch.cat(x, dim=-1).permute(0, 2, 1))
        else:
            if dates_embeddings is None:
                x = [inputs, gfs_inputs]
            else:
                x = [inputs, gfs_inputs, dates_embeddings[0], dates_embeddings[1]]
            x = self.tcn(torch.cat(x, dim=-1).permute(0, 2, 1))

        if dates_embeddings is None:
            return self.linear_time_distributed(torch.cat([x.permute(0, 2, 1), gfs_targets], dim=-1)).squeeze(-1)
        else:
            return self.linear_time_distributed(torch.cat([x.permute(0, 2, 1), gfs_targets, dates_embeddings[2], dates_embeddings[3]], dim=-1)).squeeze(-1)

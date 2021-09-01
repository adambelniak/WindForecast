from pytorch_lightning import LightningModule
import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TCNWithCNNModel(LightningModule):
    def __init__(self, cfg: Config):
        super(TCNWithCNNModel, self).__init__()
        self.cfg = cfg
        tcn_cnn_ff_input_dim = cfg.experiment.tcn_cnn_ff_input_dim
        cnn_channels = len(process_config(cfg.experiment.train_parameters_config_file))
        cnn_layers = []

        for index, filters in enumerate(cfg.experiment.cnn_filters):
            cnn_layers.append(
                nn.Conv2d(in_channels=cnn_channels, out_channels=filters, kernel_size=(3, 3), padding=(1, 1)), )
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.BatchNorm2d(num_features=filters))
            if index != len(cfg.experiment.cnn_filters) - 1:
                cnn_layers.append(nn.MaxPool2d(padding=(1, 1), kernel_size=(2, 2)))
                cnn_layers.append(nn.Dropout(cfg.experiment.dropout))
            cnn_channels = filters

        cnn_layers.append(nn.Flatten())

        tcn_layers = []
        tcn_channels = cfg.experiment.tcn_channels
        tcn_levels = len(tcn_channels)
        kernel_size = cfg.experiment.tcn_kernel_size

        for i in range(tcn_levels):
            dilation_size = 2 ** i
            in_channels = cfg.experiment.tcn_input_features if i == 0 else tcn_channels[i - 1]
            out_channels = tcn_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size)]

        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.tcn_layers = nn.Sequential(*tcn_layers)
        self.ff = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=tcn_cnn_ff_input_dim, out_features=512),
                                nn.ReLU(),
                                nn.Linear(in_features=512, out_features=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = TimeDistributed(self.cnn_layers, batch_first=True)(x)
        x = self.tcn_layers(x.permute(0, 2, 1)).squeeze()
        return self.ff(x)

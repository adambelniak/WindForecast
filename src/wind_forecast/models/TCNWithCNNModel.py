import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TCNWithCNNModel(LightningModule):
    def __init__(self, cfg: Config):
        super(TCNWithCNNModel, self).__init__()
        self.cfg = cfg
        cnn_lin_tcn_in_features = cfg.experiment.cnn_lin_tcn_in_features

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

        self.cnn = TimeDistributed(CMAXEncoder(cfg), batch_first=True)
        self.tcn_layers = nn.Sequential(*tcn_layers)
        self.ff = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=cnn_lin_tcn_in_features, out_features=512),
                                nn.ReLU(),
                                nn.Linear(in_features=512, out_features=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.cnn(x)
        x = self.tcn_layers(x.permute(0, 2, 1)).squeeze()
        return self.ff(x)

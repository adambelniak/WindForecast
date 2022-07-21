from pytorch_lightning import LightningModule
import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.util.config import process_config


class CNNModel(LightningModule):
    def __init__(self, cfg: Config):
        super(CNNModel, self).__init__()
        self.cfg = cfg
        ff_input_dim = cfg.experiment.cnn_ff_input_dim
        channels = len(process_config(cfg.experiment.train_parameters_config_file))
        cnn_layers = []

        for index, filters in enumerate(cfg.experiment.cnn_filters):
            cnn_layers.append(
                nn.Conv2d(in_channels=channels, out_channels=filters, kernel_size=(3, 3), padding=(1, 1)), )
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.BatchNorm2d(num_features=filters))
            if index != len(cfg.experiment.cnn_filters) - 1:
                cnn_layers.append(nn.MaxPool2d(padding=(1, 1), kernel_size=(2, 2)))
                cnn_layers.append(nn.Dropout(cfg.experiment.dropout))
            channels = filters

        self.model = nn.Sequential(*cnn_layers,
                                   nn.Flatten(),
                                   nn.Linear(in_features=ff_input_dim[0], out_features=ff_input_dim[1]),
                                   nn.ReLU(),
                                   nn.Linear(in_features=ff_input_dim[1], out_features=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

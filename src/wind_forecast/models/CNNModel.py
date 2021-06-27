from pytorch_lightning import LightningModule
import torch
from torch import nn

from wind_forecast.config.register import Config


class CNNModel(LightningModule):
    def __init__(self, cfg: Config):
        super(CNNModel, self).__init__()
        self.cfg = cfg
        channels, width, height = cfg.experiment.input_size
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(padding=(1, 1), kernel_size=(2, 2)),
            nn.Dropout(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(padding=(1, 1), kernel_size=(2, 2)),
            nn.Dropout(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(padding=(1, 1), kernel_size=(2, 2)),
            nn.Dropout(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(padding=(1, 1), kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=7680, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

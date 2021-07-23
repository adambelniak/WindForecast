from pytorch_lightning import LightningModule
import torch
from torch import nn

from wind_forecast.config.register import Config


class DenseModel(LightningModule):
    def __init__(self, cfg: Config):
        super(DenseModel, self).__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Linear(in_features=cfg.experiment.input_size, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=2048),
            nn.Linear(in_features=2048, out_features=8192),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=8192),
            nn.Linear(in_features=8192, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

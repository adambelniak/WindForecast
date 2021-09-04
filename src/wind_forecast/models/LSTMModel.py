import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config


class LSTMModel(LightningModule):

    def __init__(self, config: Config):
        super(LSTMModel, self).__init__()
        input_size = len(config.experiment.synop_train_features)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=4*input_size*input_size, batch_first=True, dropout=0.5)
        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=4*input_size*input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        out = self.dense(output[:, -1, :])
        return torch.squeeze(out)

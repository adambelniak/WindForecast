import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config


class LSTMModel(LightningModule):

    def __init__(self, config: Config):
        super(LSTMModel, self).__init__()
        self.config = config
        input_size = len(config.experiment.lstm_train_parameters)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=4*input_size*input_size, batch_first=True, dropout=0.2)
        self.dense = nn.Sequential(
            nn.Linear(in_features=4*input_size*input_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        out = self.dense(output[:, -1, :])
        return out.reshape((out.shape[0]))

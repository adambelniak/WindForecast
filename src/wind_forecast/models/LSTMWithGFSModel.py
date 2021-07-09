import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config


class LSTMModel(LightningModule):

    def __init__(self, config: Config):
        super(LSTMModel, self).__init__()
        self.config = config
        input_size = config.experiment.input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=4*input_size*input_size, batch_first=True, dropout=0.5)
        self.dense = nn.Sequential(
            nn.Linear(in_features=4*input_size*input_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, synop_input, gfs_input):
        lstm_out, _ = self.lstm(synop_input)
        combined = torch.cat((lstm_out[:, -1, :], gfs_input), dim=1)
        out = self.dense(combined)
        return out.reshape((out.shape[0]))

import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config


class StackedLSTMWithGFSModel(LightningModule):

    def __init__(self, config: Config):
        super(StackedLSTMWithGFSModel, self).__init__()
        self.config = config
        input_size = len(config.experiment.synop_train_features)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=4*input_size*input_size, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=4*input_size*input_size, hidden_size=70, batch_first=True, dropout=0.5)
        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=71, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, synop_input, gfs_input) -> torch.Tensor:
        lstm_out, _ = self.lstm(synop_input)
        lstm_out, _ = self.lstm2(lstm_out)
        combined = torch.cat((lstm_out[:, -1, :], torch.unsqueeze(gfs_input, -1)), dim=1)
        out = self.dense(combined)
        return torch.squeeze(out)

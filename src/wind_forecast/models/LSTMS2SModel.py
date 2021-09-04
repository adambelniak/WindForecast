import math

import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class LSTMS2SModel(LightningModule):

    def __init__(self, config: Config):
        super(LSTMS2SModel, self).__init__()
        input_size = len(config.experiment.synop_train_features)
        self.sequence_length = config.experiment.sequence_length
        self.teacher_forcing_epoch_num = config.experiment.teacher_forcing_epoch_num
        self.gradual_teacher_forcing = config.experiment.gradual_teacher_forcing
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=4*input_size*input_size, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=4*input_size*input_size, hidden_size=input_size, batch_first=True, dropout=0.5)
        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=input_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epoch: int, stage=None) -> torch.Tensor:
        output, _ = self.lstm1(inputs)
        output, _ = self.lstm2(output)
        if epoch < self.teacher_forcing_epoch_num and stage in [None, 'fit']:
            # Teacher forcing
            pred = output[:, -1:, :] # first in pred sequence
            if self.gradual_teacher_forcing:
                targets_shifted = targets[:, :-1, ]
                first_taught = math.floor(epoch / self.teacher_forcing_epoch_num * self.sequence_length)
                for frame in range(first_taught - 1): # do normal prediction for the beginning frames
                    next_pred, _ = self.lstm1(pred[:, -1:, :])
                    next_pred, _ = self.lstm2(next_pred)
                    pred = torch.cat([pred, next_pred], 1)

                # then, do teacher forcing
                next_pred, _ = self.lstm1(targets_shifted[:, first_taught:, :])
                next_pred, _ = self.lstm2(next_pred)
                pred = torch.cat([pred, next_pred], 1)

            else: # non-gradual, just basic teacher forcing
                targets_shifted = torch.cat([pred, targets], 1)[:, :-1, ]
                pred, _ = self.lstm1(targets_shifted)
                pred, _ = self.lstm2(pred)

        else:
            # inference
            pred = output[:, -1:, :]
            for frame in range(inputs.size(1) - 1):
                next_pred, _ = self.lstm1(pred[:, -1:, :])
                next_pred, _ = self.lstm2(next_pred)
                pred = torch.cat([pred, next_pred], 1)

        return torch.squeeze(TimeDistributed(self.dense, batch_first=True)(pred))

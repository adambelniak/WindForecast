from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TCNS2SFeatureSeparableModel(LightningModule):
    def __init__(self, config: Config):
        super(TCNS2SFeatureSeparableModel, self).__init__()
        self.config = config
        self.synop_train_features_length = len(config.experiment.synop_train_features)
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3

        dnn_features = num_channels[-1]

        self.sep_tcns = []
        self.sep_linears = []
        for first_feat in range(0, self.synop_train_features_length - 1):
            for second_feat in range(first_feat + 1, self.synop_train_features_length):
                tcn_dnn_layers = []
                for i in range(num_levels):
                    dilation_size = 2 ** i
                    # 4/2 input channels - a pair of synop variables +/- dates
                    in_channels = (4 if config.experiment.with_dates_inputs else 2) if i == 0 else num_channels[i - 1]
                    out_channels = num_channels[i]
                    tcn_dnn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                                     padding=(kernel_size - 1) * dilation_size, dropout=config.experiment.dropout)]
                self.sep_tcns.append(nn.Sequential(*tcn_dnn_layers))
                self.sep_linears.append(TimeDistributed(nn.Sequential(nn.Linear(in_features=dnn_features, out_features=2), nn.ReLU()), batch_first=True))

        self.sep_tcns = nn.ModuleList(self.sep_tcns)
        self.sep_linears = nn.ModuleList(self.sep_linears)

        head_linear_features = self.synop_train_features_length * (self.synop_train_features_length - 1)
        linear = nn.Sequential(
            nn.Linear(in_features=head_linear_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        outputs = []
        for first_feat in range(0, self.synop_train_features_length - 1):
            for second_feat in range(first_feat + 1, self.synop_train_features_length):
                if self.config.experiment.with_dates_inputs:
                    inputs = torch.cat([synop_inputs[:, :, first_feat:first_feat+1], synop_inputs[:, :, second_feat:second_feat+1], dates_embedding[0], dates_embedding[1]], -1)
                else:
                    inputs = torch.cat([synop_inputs[:, :, first_feat:first_feat+1], synop_inputs[:, :, second_feat:second_feat+1]], -1)
                output = self.sep_tcns[first_feat](inputs.permute(0, 2, 1)).permute(0, 2, 1)
                outputs.append(self.sep_linears[first_feat](output))

        x = torch.cat(outputs, dim=-1)

        # if self.config.experiment.with_dates_inputs:
        #     return self.linear_time_distributed(
        #         torch.cat([x.permute(0, 2, 1), dates_embedding[2], dates_embedding[3]], -1)).squeeze(-1)
        # else:
        return self.linear_time_distributed(x).squeeze(-1)

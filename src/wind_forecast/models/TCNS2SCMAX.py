from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TCNS2SCMAX(LightningModule):
    def __init__(self, config: Config):
        super(TCNS2SCMAX, self).__init__()
        self.config = config
        self.future_sequence_length = config.experiment.future_sequence_length
        self.conv_encoder = CMAXEncoder(config)

        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv, config)

        self.conv_time_distributed = TimeDistributed(self.conv, batch_first=True)

        self.cnn_lin_tcn = TimeDistributed(nn.Linear(in_features=config.experiment.cnn_lin_tcn_in_features,
                                                     out_features=config.experiment.tcn_channels[0]),
                                           batch_first=True)
        self.tcn = self.create_tcn_layers()

        if self.config.experiment.with_dates_inputs:
            features = config.experiment.tcn_channels[-1] + 6
        else:
            features = config.experiment.tcn_channels[-1]

        if self.config.experiment.use_gfs_data:
            features += 1

        linear = nn.Sequential(
            nn.Linear(in_features=features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def create_tcn_layers(self, ):
        tcn_layers = []
        tcn_channels = self.config.experiment.tcn_channels
        tcn_channels[0] += len(self.config.experiment.synop_train_features) + len(self.config.experiment.periodic_features)
        tcn_channels[0] += 6 if self.config.experiment.with_dates_inputs else 0

        if self.config.experiment.use_gfs_data and self.config.experiment.use_all_gfs_params:
            gfs_params = process_config(config.experiment.train_parameters_config_file)
            gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                gfs_params_len += 1  # V and U will be expanded int velocity, sin and cos
            tcn_channels[0] += gfs_params_len

        kernel_size = self.config.experiment.tcn_kernel_size
        for i in range(len(tcn_channels) - 1):
            dilation_size = 2 ** i
            in_channels = tcn_channels[i]
            out_channels = tcn_channels[i + 1]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size)]

        return nn.Sequential(*tcn_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()

        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]
        gfs_targets = None if self.config.experiment.use_gfs_data is False else batch[
            BatchKeys.GFS_FUTURE_Y.value].float()

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_gfs_data and self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                x = [synop_inputs, gfs_inputs, *dates_embedding[0]]
            else:
                x = [synop_inputs, *dates_embedding[0]]
        else:
            if self.config.experiment.use_gfs_data and self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                x = [synop_inputs, gfs_inputs]
            else:
                x = [synop_inputs]

        cmax_embedding = self.cnn(cmax_inputs.unsqueeze(2))
        cmax_embedding = self.cnn_lin_tcn(cmax_embedding)
        x = torch.cat([*x, cmax_embedding], dim=-1)
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        mem = x[:, -self.future_sequence_length:, :]

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_gfs_data:
                return self.linear_time_distributed(torch.cat([mem, gfs_targets, *dates_embedding[1]], -1)).squeeze(-1)
            return self.linear_time_distributed(torch.cat([mem, *dates_embedding[1]], -1)).squeeze(-1)
        else:
            if self.config.experiment.use_gfs_data:
                return self.linear_time_distributed(torch.cat([mem, gfs_targets], -1)).squeeze(-1)
            return self.linear_time_distributed(mem).squeeze(-1)

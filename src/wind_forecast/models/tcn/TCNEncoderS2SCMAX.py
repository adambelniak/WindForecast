from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.tcn.TCNEncoder import TemporalBlock
from wind_forecast.models.tcn.TCNEncoderS2S import TCNEncoderS2S
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TCNEncoderS2SCMAX(TCNEncoderS2S):
    def __init__(self, config: Config):
        super().__init__(config)
        self.conv_encoder = CMAXEncoder(config)
        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv, config)

        self.conv_time_distributed = TimeDistributed(self.conv, batch_first=True)

        self.cnn_lin_tcn = TimeDistributed(nn.Linear(in_features=config.experiment.cnn_lin_tcn_in_features,
                                                     out_features=config.experiment.tcn_channels[0]),
                                           batch_first=True)
        self.tcn = self.create_tcn_layers()

        features = config.experiment.tcn_channels[-1]

        if self.config.experiment.use_gfs_data:
            features += 1

        linear = nn.Sequential(
            nn.Linear(in_features=features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def create_tcn_layers(self):
        tcn_layers = []
        tcn_channels = self.config.experiment.tcn_channels
        tcn_channels[0] += len(self.config.experiment.synop_train_features) + len(self.config.experiment.synop_periodic_features)

        if self.config.experiment.with_dates_inputs:
            tcn_channels[0] += 2 if not self.use_time2vec else 2 * self.time2vec_embedding_size

        if self.config.experiment.use_gfs_data and self.config.experiment.use_all_gfs_params:
            gfs_params = process_config(self.config.experiment.train_parameters_config_file)
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
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         self.use_gfs, False)
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()

        if self.use_gfs:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()

        cmax_embedding = self.cnn(cmax_inputs.unsqueeze(2))
        cmax_embedding = self.cnn_lin_tcn(cmax_embedding)
        x = torch.cat([input_elements, cmax_embedding], dim=-1)
        x = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        mem = x[:, -self.future_sequence_length:, :]

        if self.use_gfs:
            return self.classification_head(torch.cat([mem, gfs_targets], -1)).squeeze(-1)
        return self.classification_head(mem).squeeze(-1)

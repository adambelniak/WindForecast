from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TCNModel import TemporalBlock
from wind_forecast.models.Transformer import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TemporalConvNetS2SWithGFS(LightningModule):
    def __init__(self, config: Config):
        super(TemporalConvNetS2SWithGFS, self).__init__()
        self.config = config
        self.features_length = len(config.experiment.synop_train_features)
        if config.experiment.with_dates_inputs:
            self.features_length += 4
        if config.experiment.use_all_gfs_params:
            gfs_params_len = len(process_config(config.experiment.train_parameters_config_file))
            self.features_length += gfs_params_len

        self.embed_dim = self.features_length * (config.experiment.time2vec_embedding_size + 1)

        self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(self.features_length,
                                                                    config.experiment.time2vec_embedding_size),
                                                           batch_first=True)

        tcn_layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        in_channels = self.embed_dim

        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=config.experiment.dropout)]
            in_channels = num_channels[i]

        if config.experiment.with_dates_inputs:
            in_features = num_channels[-1] + 5
        else:
            in_features = num_channels[-1] + 1

        linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1)
        )
        self.tcn = nn.Sequential(*tcn_layers)
        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        gfs_targets = batch[BatchKeys.GFS_TARGETS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs, *dates_embedding[0], *dates_embedding[1]]
            else:
                x = [synop_inputs, *dates_embedding[0], *dates_embedding[1]]
        else:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs]
            else:
                x = [synop_inputs]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)
        x = self.tcn(whole_input_embedding.permute(0, 2, 1)).permute(0, 2, 1)

        if self.config.experiment.with_dates_inputs:
            return self.linear_time_distributed(torch.cat([x, gfs_targets, *dates_embedding[2], *dates_embedding[3]], -1)).squeeze(-1)
        else:
            return self.linear_time_distributed(torch.cat([x, gfs_targets], -1)).squeeze(-1)

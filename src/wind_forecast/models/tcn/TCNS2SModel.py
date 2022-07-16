from typing import Dict

import torch
import torch.nn as nn
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.tcn.TCNModel import TemporalBlock
from wind_forecast.models.decomposeable.Decomposeable import EMDDecomposeable
from wind_forecast.models.transformer.Transformer import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TemporalConvNetS2S(EMDDecomposeable):
    def __init__(self, config: Config):
        super(TemporalConvNetS2S, self).__init__(config.experiment.emd_decompose_trials)
        self.config = config
        self.use_gfs = config.experiment.use_gfs_data
        self.use_gfs_on_inputs = self.use_gfs and config.experiment.use_all_gfs_params
        self.future_sequence_length = config.experiment.future_sequence_length
        self.self_output_test = config.experiment.self_output_test
        self.use_time2vec = config.experiment.use_time2vec
        self.time2vec_embedding_size = config.experiment.time2vec_embedding_size

        self.features_length = len(config.experiment.synop_train_features) + len(config.experiment.synop_periodic_features)
        if config.experiment.with_dates_inputs:
            self.features_length += 2 if not self.use_time2vec else 2 * self.time2vec_embedding_size

        if self.use_time2vec:
            self.time_embed = TimeDistributed(Time2Vec(2, self.time2vec_embedding_size), batch_first=True)
        if self.use_gfs_on_inputs:
            gfs_params = process_config(config.experiment.train_parameters_config_file)
            gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                gfs_params_len += 1  # V and U will be expanded int velocity, sin and cos
            self.features_length += gfs_params_len

        tcn_layers = []
        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        in_channels = 1 if self.self_output_test or self.config.experiment.emd_decompose else self.features_length

        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=config.experiment.dropout)]
            in_channels = out_channels

        in_features = num_channels[-1]

        if self.use_gfs and not self.self_output_test and not self.config.experiment.emd_decompose:
            in_features += 1

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
        if self.self_output_test:
            return self.self_forward(batch, epoch, stage)

        if self.config.experiment.emd_decompose:
            synop_inputs = batch[BatchKeys.SYNOP_PAST_Y.value].float().unsqueeze(-1)
            decomposed_batch = self.decompose(synop_inputs)
            x = self.compose([self.tcn(decomposed.permute(0, 2, 1)).permute(0, 2, 1) for decomposed in decomposed_batch])
            x = x[:, -self.future_sequence_length:, :]
            return self.linear_time_distributed(x).squeeze(-1)

        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()

        dates = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]
        gfs_targets = None if not self.use_gfs else batch[BatchKeys.GFS_FUTURE_Y.value].float()

        if self.config.experiment.with_dates_inputs:
            dates_embedding = dates[0]
            if self.use_time2vec:
                dates_embedding = self.time_embed(dates[0])

            if self.use_gfs_on_inputs:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                x = [synop_inputs, gfs_inputs, dates_embedding]
            else:
                x = [synop_inputs, dates_embedding]
        else:
            if self.use_gfs_on_inputs:
                gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
                x = [synop_inputs, gfs_inputs]
            else:
                x = [synop_inputs]

        x = self.tcn(torch.cat(x, dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        x = x[:, -self.future_sequence_length:, :]

        if self.use_gfs:
            return self.linear_time_distributed(torch.cat([x, gfs_targets], -1)).squeeze(-1)
        return self.linear_time_distributed(x).squeeze(-1)

    def self_forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_targets = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().unsqueeze(-1)

        x = self.tcn(synop_targets.permute(0, 2, 1))
        x = x[:, :, -self.future_sequence_length:]

        return self.linear_time_distributed(x.permute(0, 2, 1)).squeeze(-1)

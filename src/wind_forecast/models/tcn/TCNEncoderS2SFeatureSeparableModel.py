from typing import Dict

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.tcn.TCNEncoder import TemporalBlock
from wind_forecast.models.transformer.Transformer import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TCNEncoderS2SFeatureSeparableModel(LightningModule):
    def __init__(self, config: Config):
        super(TCNEncoderS2SFeatureSeparableModel, self).__init__()
        self.config = config
        self.use_gfs = config.experiment.use_gfs_data
        self.use_gfs_on_input = self.use_gfs and config.experiment.use_all_gfs_params
        self.future_sequence_length = config.experiment.future_sequence_length
        self.synop_train_features_length = len(config.experiment.synop_train_features) + len(config.experiment.synop_periodic_features)
        self.use_time2vec = config.experiment.use_time2vec
        self.time2vec_embedding_size = config.experiment.time2vec_embedding_size

        if self.use_time2vec:
            self.time_embed = TimeDistributed(Time2Vec(2, self.time2vec_embedding_size), batch_first=True)

        num_channels = config.experiment.tcn_channels
        num_levels = len(num_channels)
        kernel_size = 3
        if self.use_gfs_on_input:
            gfs_params = process_config(config.experiment.train_parameters_config_file)
            gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                gfs_params_len += 1  # V and U will be expanded int velocity, sin and cos
            self.tcn_features = self.synop_train_features_length + gfs_params_len
        else:
            self.tcn_features = self.synop_train_features_length

        dnn_features = num_channels[-1]

        self.sep_tcns = []
        self.sep_linears = []
        for first_feat in range(0, self.tcn_features - 1):
            for second_feat in range(first_feat + 1, self.tcn_features):
                tcn_dnn_layers = []
                for i in range(num_levels):
                    dilation_size = 2 ** i
                    # 2 input channels - a pair of synop variables + time variables
                    in_channels = (2 if not self.config.experiment.with_dates_inputs else
                                   (4 if not self.use_time2vec else 2 + 2 * self.time2vec_embedding_size)) if i == 0 else\
                        num_channels[i - 1]
                    out_channels = num_channels[i]
                    tcn_dnn_layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size,
                                                     padding=(kernel_size - 1) * dilation_size, dropout=config.experiment.dropout)]
                self.sep_tcns.append(nn.Sequential(*tcn_dnn_layers))
                self.sep_linears.append(TimeDistributed(nn.Sequential(nn.Linear(in_features=dnn_features, out_features=2), nn.ReLU()), batch_first=True))

        self.sep_tcns = nn.ModuleList(self.sep_tcns)
        self.sep_linears = nn.ModuleList(self.sep_linears)

        head_linear_features = self.tcn_features * (self.tcn_features - 1)

        if self.use_gfs:
            head_linear_features += 1
        linear = nn.Sequential(
            nn.Linear(in_features=head_linear_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

        self.linear_time_distributed = TimeDistributed(linear, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        dates = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]
        gfs_targets = None if not self.use_gfs else batch[BatchKeys.GFS_FUTURE_Y.value].float()

        if self.use_gfs_on_input:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            all_inputs = torch.cat([synop_inputs, gfs_inputs], -1)
        else:
            all_inputs = synop_inputs

        outputs = []
        for first_feat in range(0, self.tcn_features - 1):
            for second_feat in range(first_feat + 1, self.tcn_features):
                if self.config.experiment.with_dates_inputs:
                    dates_embedding = dates[0]
                    if self.use_time2vec:
                        dates_embedding = self.time_embed(dates[0])
                    inputs = torch.cat([all_inputs[:, :, first_feat:first_feat+1], all_inputs[:, :, second_feat:second_feat+1],
                                        dates_embedding], -1)
                else:
                    inputs = torch.cat([all_inputs[:, :, first_feat:first_feat+1], all_inputs[:, :, second_feat:second_feat+1]], -1)
                output = self.sep_tcns[first_feat](inputs.permute(0, 2, 1)).permute(0, 2, 1)
                outputs.append(self.sep_linears[first_feat](output)[:, -self.future_sequence_length:, :])

        if self.use_gfs:
            return self.linear_time_distributed(torch.cat([*outputs, gfs_targets], -1)).squeeze(-1)
        return self.linear_time_distributed(torch.cat(outputs, -1)).squeeze(-1)

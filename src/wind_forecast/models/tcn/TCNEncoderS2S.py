from typing import Dict

import torch
import torch.nn as nn
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.decomposeable.Decomposeable import EMDDecomposeable
from wind_forecast.models.value2vec.Value2Vec import Value2Vec
from wind_forecast.models.tcn.TCNEncoder import TemporalBlock
from wind_forecast.models.time2vec.Time2Vec import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.common_util import get_pretrained_artifact_path, get_pretrained_state_dict
from wind_forecast.util.config import process_config


# EMD not examined so far
class TCNEncoderS2S(EMDDecomposeable):
    def __init__(self, config: Config):
        super(TCNEncoderS2S, self).__init__(config.experiment.emd_decompose_trials)
        self.config = config
        self.dropout = config.experiment.dropout
        self.use_gfs = config.experiment.use_gfs_data
        self.gfs_on_head = config.experiment.gfs_on_head
        self.future_sequence_length = config.experiment.future_sequence_length
        self.self_output_test = config.experiment.self_output_test
        self.tcn_channels = config.experiment.tcn_channels
        self.num_layers = len(self.tcn_channels)
        self.kernel_size = config.experiment.tcn_kernel_size

        self.features_length = len(config.experiment.synop_train_features) + len(config.experiment.synop_periodic_features)
        if self.config.experiment.stl_decompose:
            self.features_length *= 3
            self.features_length += 1  # + 1 for non-decomposed target param
        self.time2vec_embedding_factor = config.experiment.time2vec_embedding_factor
        self.value2vec_embedding_factor = config.experiment.value2vec_embedding_factor
        self.use_time2vec = config.experiment.use_time2vec and config.experiment.with_dates_inputs
        self.use_value2vec = config.experiment.use_value2vec and self.value2vec_embedding_factor > 0

        if not self.use_value2vec:
            self.value2vec_embedding_factor = 0

        if self.use_gfs:
            gfs_params = process_config(config.experiment.train_parameters_config_file).params
            self.gfs_params_len = len(gfs_params)
            param_names = [x['name'] for x in gfs_params]
            if "V GRD" in param_names and "U GRD" in param_names:
                self.gfs_params_len += 1  # V and U will be expanded int velocity, sin and cos
            if config.experiment.stl_decompose:
                self.gfs_params_len = 3 * self.gfs_params_len + 1
            self.features_length += self.gfs_params_len

        if self.use_time2vec and self.time2vec_embedding_factor == 0:
            self.time2vec_embedding_factor = self.features_length

        self.dates_dim = self.config.experiment.dates_tensor_size * self.time2vec_embedding_factor if self.use_time2vec \
            else 2 * self.config.experiment.dates_tensor_size

        if self.use_time2vec:
            self.time_embed = TimeDistributed(Time2Vec(self.config.experiment.dates_tensor_size,
                                                        self.time2vec_embedding_factor), batch_first=True)

        if self.use_value2vec:
            self.value_embed = TimeDistributed(Value2Vec(self.features_length, self.value2vec_embedding_factor),
                                               batch_first=True)

        self.embed_dim = self.features_length * (self.value2vec_embedding_factor + 1)
        if config.experiment.with_dates_inputs:
            self.embed_dim += self.dates_dim

        self.create_tcn_encoder()

        self.regression_head_features = self.tcn_channels[-1]
        if self.use_gfs:
            self.regression_head_features += 1
        self.create_regression_head()

        if config.experiment.use_pretrained_artifact and type(self).__name__ is "TCNEncoderS2S":
            pretrained_autoencoder_path = get_pretrained_artifact_path(config.experiment.pretrained_artifact)
            self.load_state_dict(get_pretrained_state_dict(pretrained_autoencoder_path))
            return

    def create_regression_head(self):
        dense_layers = []
        features = self.regression_head_features
        for neurons in self.config.experiment.regressor_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))

        self.regressor_head = nn.Sequential(*dense_layers)

    def create_tcn_encoder(self):
        tcn_layers = []
        in_channels = 1 if self.self_output_test or self.config.experiment.emd_decompose else self.embed_dim

        for i in range(self.num_layers):
            dilation_size = 2 ** i
            out_channels = self.tcn_channels[i]
            tcn_layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, dilation=dilation_size,
                                         padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
            in_channels = out_channels

        self.encoder = nn.Sequential(*tcn_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        if self.self_output_test:
            return self.self_forward(batch, epoch, stage)

        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         self.use_gfs, False)
        x = self.encoder(input_elements.permute(0, 2, 1)).permute(0, 2, 1)
        x = x[:, -self.future_sequence_length:, :]

        if self.use_gfs and self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return self.regressor_head(torch.cat([x, gfs_targets], -1)).squeeze(-1)
        return self.regressor_head(x).squeeze(-1)

    def self_forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_targets = batch[BatchKeys.SYNOP_FUTURE_Y.value].float().unsqueeze(-1)

        x = self.encoder(synop_targets.permute(0, 2, 1))
        x = x[:, :, -self.future_sequence_length:]

        return self.regressor_head(x.permute(0, 2, 1)).squeeze(-1)

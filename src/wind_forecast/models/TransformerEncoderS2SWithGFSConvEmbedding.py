from typing import Dict

from torch import nn
import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import PositionalEncoding, TransformerBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TransformerEncoderS2SWithGFSConvEmbedding(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        if config.experiment.use_all_gfs_params:
            gfs_params_len = len(process_config(config.experiment.train_parameters_config_file))
            self.features_length += gfs_params_len
        self.embed_dim = config.experiment.conv_embedding_dim

        self.time_2_vec_time_distributed = nn.Conv1d(in_channels=self.features_length,
                                                     out_channels=self.embed_dim,
                                                     kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim,
                                                   dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers,
                                             encoder_norm)
        dense_layers = []
        features = self.embed_dim + 1

        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        gfs_targets = batch[BatchKeys.GFS_TARGETS.value].float()
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_EMBEDDING.value]

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

        whole_input_embedding = self.time_2_vec_time_distributed(torch.cat(x, -1).permute(0, 2, 1)).permute(0, 2, 1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        x = self.encoder(x)

        return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, gfs_targets], -1)), -1)
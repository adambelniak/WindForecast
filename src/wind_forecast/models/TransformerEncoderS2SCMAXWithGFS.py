import math
from typing import Dict

import torch
import torch.nn as nn
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.Transformer import TransformerEncoderGFSBaseProps, PositionalEncoding
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2SCMAXWithGFS(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        out_channels = config.experiment.cnn_filters[-1]
        self.conv = CMAXEncoder(config)
        for _ in config.experiment.cnn_filters:
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)

        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv, config)
        self.conv_time_distributed = TimeDistributed(self.conv, batch_first=True)

        self.embed_dim += conv_W * conv_H * out_channels

        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.ff_dim,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.transformer_layers_num, encoder_norm)

        dense_layers = []
        features = self.embed_dim + 1
        if self.config.experiment.with_dates_inputs:
            features += 6
        for neurons in self.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()

        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        if self.config.experiment.use_all_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            x = [synop_inputs, gfs_inputs]
        else:
            x = [synop_inputs]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1)), cmax_embeddings], -1)
        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        output = self.encoder(x)
        output = output[:, -self.future_sequence_length:, :]

        if self.config.experiment.with_dates_inputs:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets, *dates_tensors[1]], -1)), -1)
        else:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets], -1)), -1)


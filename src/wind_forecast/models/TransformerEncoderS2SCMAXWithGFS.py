import math
from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import TransformerBaseProps, PositionalEncoding, Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TransformerEncoderS2SCMAXWithGFS(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        conv_layers = []
        in_channels = 1
        for index, filters in enumerate(config.experiment.cnn_filters):
            out_channels = filters
            conv_layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=out_channels),
            ])
            if index != len(config.experiment.cnn_filters) - 1:
                conv_layers.append(nn.Dropout(config.experiment.dropout))
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers, nn.Flatten(),
                                  nn.Linear(in_features=conv_W * conv_H * out_channels, out_features=conv_W * conv_H * out_channels))
        self.conv_time_distributed = TimeDistributed(self.conv)

        self.embed_dim = self.features_len * (config.experiment.time2vec_embedding_size + 1) + conv_W * conv_H * out_channels
        if config.experiment.use_all_gfs_params:
            self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(self.features_len + len(process_config(config.experiment.train_parameters_config_file)),
                                                                        config.experiment.time2vec_embedding_size), batch_first=True)
            self.embed_dim += len(process_config(config.experiment.train_parameters_config_file)) * (config.experiment.time2vec_embedding_size + 1)

        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)

        dense_layers = []
        if config.experiment.with_dates_inputs:
            features = self.embed_dim + 3
        else:
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
        dates_embedding = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_EMBEDDING.value]
        cmax_inputs = batch[BatchKeys.CMAX_INPUTS.value].float()

        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        if self.config.experiment.with_dates_inputs:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs, dates_embedding[0], dates_embedding[1]]
            else:
                x = [synop_inputs, dates_embedding[0], dates_embedding[1]]
        else:
            if self.config.experiment.use_all_gfs_params:
                gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
                x = [synop_inputs, gfs_inputs]
            else:
                x = [synop_inputs]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1)), cmax_embeddings], -1)
        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        output = self.encoder(x)

        if self.config.experiment.with_dates_inputs:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets, dates_embedding[2], dates_embedding[3]], -1)), -1)
        else:
            return torch.squeeze(self.classification_head_time_distributed(output, gfs_targets), -1)


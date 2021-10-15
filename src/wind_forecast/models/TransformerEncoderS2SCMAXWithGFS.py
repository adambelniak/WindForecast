import math

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


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

        self.conv = nn.Sequential(*conv_layers, nn.Flatten())
        self.conv_time_distributed = TimeDistributed(self.conv)

        self.embed_dim = self.features_len * (config.experiment.time2vec_embedding_size + 1) + conv_W * conv_H * out_channels

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)
        self.linear = nn.Linear(in_features=self.embed_dim + 1, out_features=1)
        self.linear_time_distributed = TimeDistributed(self.linear, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, inputs, gfs_input, cmax_inputs, targets: torch.Tensor, epoch: int, stage=None):
        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))
        time_embedding = self.time2vec_time_distributed(inputs)
        x = torch.cat([inputs, time_embedding, cmax_embeddings], -1)
        x = self.pos_encoder(x) if self.use_pos_encoding else x
        x = self.encoder(x)

        return torch.squeeze(self.linear_time_distributed(torch.cat([x, gfs_input], dim=-1)), dim=-1)


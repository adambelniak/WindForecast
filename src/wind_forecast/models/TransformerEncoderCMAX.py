import math

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderCMAX(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w

        conv_layers = []
        in_channels = 1
        for filters in config.experiment.cnn_filters:
            out_channels = filters
            conv_layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3)),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=out_channels),
                nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1))
            ])
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers, nn.Flatten())
        self.conv_time_distributed = TimeDistributed(self.conv)
        self.embed_dim = self.features_len * (config.experiment.time2vec_embedding_size + 1) + conv_W * conv_H * out_channels

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)
        self.linear = nn.Linear(in_features=self.embed_dim * config.experiment.sequence_length, out_features=1)
        self.flatten = nn.Flatten()

    def forward(self, inputs, cmax_inputs):
        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))
        time_embedding = self.time2vec_time_distributed(inputs)
        x = torch.cat([inputs, time_embedding, cmax_embeddings], -1)
        x = self.pos_encoder(x) if self.use_pos_encoding else x
        x = self.encoder(x)
        x = self.flatten(x)  # flat vector of features out

        return torch.squeeze(self.linear(x), dim=-1)


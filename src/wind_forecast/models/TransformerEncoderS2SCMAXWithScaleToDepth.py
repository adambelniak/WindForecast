import math
import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps, PositionalEncoding
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2SCMAXWithScaleToDepth(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.scaling_factor = config.experiment.STD_scaling_factor
        conv_H = config.experiment.cmax_h // self.scaling_factor
        conv_W = config.experiment.cmax_w // self.scaling_factor
        conv_layers = []
        in_channels = pow(self.scaling_factor, 2)
        for index, filters in enumerate(config.experiment.cnn_filters):
            out_channels = filters
            conv_layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=out_channels),
            ])
            if index != len(config.experiment.cnn_filters) - 1:
                conv_layers.append(nn.Dropout(self.dropout))
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers, nn.Flatten(),
                                  nn.Linear(in_features=conv_W * conv_H * out_channels, out_features=conv_W * conv_H * out_channels))
        self.conv_time_distributed = TimeDistributed(self.conv)
        self.embed_dim = self.features_len * (config.experiment.time2vec_embedding_size + 1) + conv_W * conv_H * out_channels
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)

    def forward(self, inputs, cmax_inputs, targets: torch.Tensor, epoch: int, stage=None):
        cmax_inputs = cmax_inputs.unsqueeze(2)
        cmax_reshape = cmax_inputs.contiguous().view(cmax_inputs.size()[0] * cmax_inputs.size()[1], *cmax_inputs.size()[2:])
        cmax = torch.nn.functional.pixel_unshuffle(cmax_reshape, self.scaling_factor)
        cmax = cmax.contiguous().view(cmax_inputs.size(0), cmax_inputs.size(1), *cmax.size()[1:])

        cmax_embeddings = self.conv_time_distributed(cmax)
        time_embedding = self.time_2_vec_time_distributed(inputs)
        x = torch.cat([inputs, time_embedding, cmax_embeddings], -1)
        x = self.pos_encoder(x) if self.use_pos_encoding else x
        x = self.encoder(x)

        return torch.squeeze(self.classification_head_time_distributed(x), dim=-1)

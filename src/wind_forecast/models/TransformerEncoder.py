import math

import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class Time2Vec(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.time2vec_dim = config.experiment.time2vec_embedding_size - 1
        # trend
        self.wb = nn.Parameter(data=torch.empty(size=(len(config.experiment.synop_train_features),)), requires_grad=True)
        self.bb = nn.Parameter(data=torch.empty(size=(len(config.experiment.synop_train_features),)), requires_grad=True)

        # periodic
        self.wa = nn.Parameter(data=torch.empty(size=(1, len(config.experiment.synop_train_features), self.time2vec_dim)), requires_grad=True)
        self.ba = nn.Parameter(data=torch.empty(size=(1, len(config.experiment.synop_train_features), self.time2vec_dim)), requires_grad=True)

        self.wb.data.uniform_(-1, 1)
        self.bb.data.uniform_(-1, 1)
        self.wa.data.uniform_(-1, 1)
        self.ba.data.uniform_(-1, 1)

    def forward(self, inputs):
        bias = torch.mul(self.wb, inputs) + self.bb
        dp = torch.mul(torch.unsqueeze(inputs, -1), self.wa) + self.ba
        wgts = torch.sin(dp)

        ret = torch.cat([torch.unsqueeze(bias, -1), wgts], -1)
        ret = torch.reshape(ret, (-1, inputs.shape[1] * (self.time2vec_dim + 1)))
        return ret


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, sequence_length: int = 24):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, sequence_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)[:,:d_model // 2]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = torch.cat((x, self.pe.expand(x.shape)), -1)
        return self.dropout(x)


class TransformerEncoder(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        embed_dim = len(config.experiment.synop_train_features) * (config.experiment.time2vec_embedding_size + 1)
        self.time2vec = Time2Vec(config)
        self.pos_encoder = PositionalEncoding(embed_dim, config.experiment.dropout, config.experiment.sequence_length)
        features_len = len(config.experiment.synop_train_features)
        d_model = features_len * (config.experiment.time2vec_embedding_size + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=config.experiment.dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers)
        self.linear = nn.Linear(in_features=embed_dim * config.experiment.sequence_length, out_features=1)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        time_embedding = TimeDistributed(self.time2vec, batch_first=True)(inputs)
        x = torch.cat([inputs, time_embedding], -1)
        x = self.encoder(x)
        x = self.flatten(x)  # flat vector of features out

        return torch.squeeze(self.linear(x))


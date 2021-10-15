import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2SWithGFS(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)
        self.linear = nn.Linear(in_features=self.embed_dim + 1, out_features=1)
        self.flatten = nn.Flatten()

    def forward(self, inputs, gfs_input, targets: torch.Tensor, epoch: int, stage=None):
        time_embedding = self.time_2_vec_time_distributed(inputs)
        x = torch.cat([inputs, time_embedding], -1)
        x = self.pos_encoder(x) if self.use_pos_encoding else x
        x = self.encoder(x)

        return torch.squeeze(TimeDistributed(self.linear, batch_first=True)(torch.cat([x, gfs_input], dim=-1)), dim=-1)


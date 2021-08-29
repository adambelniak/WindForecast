import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import Time2Vec, PositionalEncoding, AttentionBlock, TimeDistributed


class Transformer(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        embed_dim = len(config.experiment.synop_train_features) * (config.experiment.time2vec_embedding_size + 1)
        self.time2vec = Time2Vec(config)
        self.pos_encoder = PositionalEncoding(embed_dim, config.experiment.dropout, config.experiment.sequence_length)
        self.attention_layers = nn.Sequential(*[AttentionBlock(config) for _ in range(config.experiment.transformer_attention_layers)])
        self.linear = nn.Linear(in_features=embed_dim * config.experiment.sequence_length + 1, out_features=1)
        self.flatten = nn.Flatten()

    def forward(self, synop_input, gfs_input):
        time_embedding = TimeDistributed(self.time2vec, batch_first=True)(synop_input)
        x = torch.cat([synop_input, time_embedding], -1)
        x = self.attention_layers(x)

        x = self.flatten(x)  # flat vector of features out

        return torch.squeeze(self.linear(torch.cat((x, gfs_input.unsqueeze(-1)), dim=1)))

import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import PositionalEncoding
from wind_forecast.models.TransformerEncoder import Time2Vec, TimeDistributed


class Transformer(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        embed_dim = len(config.experiment.synop_train_features) * (config.experiment.time2vec_embedding_size + 1)
        self.time2vec = Time2Vec(len(config.experiment.synop_train_features), config.experiment.time2vec_embedding_size)
        self.pos_encoder = PositionalEncoding(embed_dim, config.experiment.dropout, config.experiment.sequence_length)
        features_len = len(config.experiment.synop_train_features)
        d_model = features_len * (config.experiment.time2vec_embedding_size + 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim,
                                                   dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers,
                                             encoder_norm)
        self.linear = nn.Linear(in_features=embed_dim * config.experiment.sequence_length + 1, out_features=1)
        self.flatten = nn.Flatten()

    def forward(self, synop_input, gfs_input):
        time_embedding = TimeDistributed(self.time2vec, batch_first=True)(synop_input)
        x = torch.cat([synop_input, time_embedding], -1)
        x = self.encoder(x)
        x = self.flatten(x)  # flat vector of features out

        return torch.squeeze(self.linear(torch.cat((x, gfs_input.unsqueeze(-1)), dim=1)))


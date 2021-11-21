import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps, Time2Vec, PositionalEncoding
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2S(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        input_features_length = self.features_len

        if config.experiment.with_dates_inputs:
            input_features_length += 2
            self.embed_dim += 2 * (config.experiment.time2vec_embedding_size + 1)

        self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(input_features_length,
                                                                    config.experiment.time2vec_embedding_size),
                                                           batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)
        self.flatten = nn.Flatten()
        dense_layers = []
        features = self.embed_dim
        if config.experiment.with_dates_inputs:
            features = self.embed_dim + 2

        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def forward(self, inputs, targets: torch.Tensor, epoch: int, stage=None,
                dates_embeddings: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) = None):
        if dates_embeddings is None:
            x = [inputs]
        else:
            x = [inputs, dates_embeddings[0], dates_embeddings[1]]
        time_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)
        x = self.pos_encoder(time_embedding) if self.use_pos_encoding else time_embedding
        x = self.encoder(x)

        return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, dates_embeddings[2], dates_embeddings[3]], dim=-1)), dim=-1)


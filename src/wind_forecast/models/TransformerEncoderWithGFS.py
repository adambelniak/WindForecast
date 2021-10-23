import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps


class Transformer(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers,
                                             encoder_norm)

        dense_layers = []
        features = self.embed_dim * config.experiment.sequence_length
        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.flatten = nn.Flatten()

    def forward(self, synop_input, gfs_input):
        time_embedding = self.time_2_vec_time_distributed(synop_input)
        x = torch.cat([synop_input, time_embedding], -1)
        x = self.pos_encoder(x) if self.use_pos_encoding else x
        x = self.encoder(x)
        x = self.flatten(x)  # flat vector of synop_features out

        return torch.squeeze(self.classification_head(torch.cat((x, gfs_input.unsqueeze(-1)), dim=1)))


import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import TransformerBaseProps, Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.config import process_config


class TransformerEncoderS2SWithGFS(TransformerBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        if config.experiment.use_all_gfs_as_input:
            self.time_2_vec_time_distributed = TimeDistributed(Time2Vec(self.features_len + len(process_config(config.experiment.train_parameters_config_file)),
                                                                        config.experiment.time2vec_embedding_size), batch_first=True)
            self.embed_dim += len(process_config(config.experiment.train_parameters_config_file)) * (config.experiment.time2vec_embedding_size + 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)
        dense_layers = []
        features = self.embed_dim + 1
        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def forward(self, inputs, gfs_inputs, gfs_targets, targets: torch.Tensor, epoch: int, stage=None):
        if gfs_inputs is None:
            time_embedding = torch.cat([inputs, self.time_2_vec_time_distributed(inputs)], dim=-1)
        else:
            time_embedding = torch.cat([inputs, gfs_inputs,
                                        self.time_2_vec_time_distributed(torch.cat([inputs, gfs_inputs], dim=-1))], dim=-1)
        x = self.pos_encoder(time_embedding) if self.use_pos_encoding else time_embedding
        x = self.encoder(x)

        return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, gfs_targets], dim=-1)), dim=-1)


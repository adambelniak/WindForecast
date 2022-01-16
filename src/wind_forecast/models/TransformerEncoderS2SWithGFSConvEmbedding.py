from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import PositionalEncoding, Simple2Vec, TransformerGFSBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


# TODO make work after refactoring
class TransformerEncoderS2SWithGFSConvEmbedding(TransformerGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.embed_dim = config.experiment.embedding_scale * self.features_length
        self.simple_2_vec_time_distributed = TimeDistributed(Simple2Vec(self.features_length, config.experiment.embedding_scale), batch_first=True)
        if config.experiment.with_dates_inputs:
            self.embed_dim += 6

        self.features_embeds = nn.Conv1d(in_channels=self.features_length, out_channels=self.conv_embed_dim,
                                                     kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim,
                                                   dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout, self.sequence_length)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers,
                                             encoder_norm)
        dense_layers = []
        features = self.embed_dim + 1

        for neurons in config.experiment.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        gfs_targets = batch[BatchKeys.GFS_TARGETS.value].float()
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.use_all_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_INPUTS.value].float()
            x = [synop_inputs, gfs_inputs]
        else:
            x = [synop_inputs]

        input_features_embeddings = self.simple_2_vec_time_distributed(torch.cat(x, -1)) #.permute(0, 2, 1)).permute(0, 2, 1)
        if self.config.experiment.with_dates_inputs:
            # dates_embeddings = self.time_2_vec_time_distributed(dates_tensors[0])
            whole_input_embedding = torch.cat([input_features_embeddings, *dates_tensors[0]], -1)
        else:
            whole_input_embedding = input_features_embeddings

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        x = self.encoder(x)

        if self.config.experiment.with_dates_inputs:
            # dates_embeddings = self.time_2_vec_time_distributed(dates_tensors[1])
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, gfs_targets, *dates_tensors[1]], -1)), -1)
        else:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, gfs_targets], -1)), -1)
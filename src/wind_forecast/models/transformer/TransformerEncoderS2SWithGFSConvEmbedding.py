from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.transformer.Transformer import PositionalEncoding, TransformerEncoderGFSBaseProps


# TODO - on hold
class TransformerEncoderS2SWithGFSConvEmbedding(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.future_sequence_length <= self.past_sequence_length, "Past sequence length can't be shorter than future sequence length for transformer encoder arch"
        self.conv_embed_dim = config.experiment.embedding_scale * self.features_length

        self.features_embeds = nn.Conv1d(in_channels=self.features_length, out_channels=self.conv_embed_dim,
                                         kernel_size=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                   nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim,
                                                   dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_encoder_layers,
                                             encoder_norm)
        dense_layers = []
        features = self.embed_dim
        for neurons in config.experiment.transformer_classification_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=self.features_length))
        self.forecaster = nn.Sequential(*dense_layers)
        if config.experiment.with_dates_inputs:
            self.classification_head = nn.Linear(self.features_length + 5, 1)
        else:
            self.classification_head = nn.Linear(self.features_length + 1, 1)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.use_all_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            x = [synop_inputs, gfs_inputs]
        else:
            x = [synop_inputs]

        whole_input_embedding = torch.cat(
            [*x, self.features_embeds((torch.cat(x, -1).permute(0, 2, 1))).permute(0, 2, 1)], -1)
        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        if self.projection is not None:
            x = self.projection(x)
        memory = self.encoder(x)
        memory = memory[:, -self.future_sequence_length:, :]

        if self.config.experiment.with_dates_inputs:
            return torch.squeeze(self.classification_head(torch.cat([self.forecaster(memory), gfs_targets, *dates_tensors[1]], -1)), -1)
        else:
            return torch.squeeze(self.classification_head(torch.cat([self.forecaster(memory), gfs_targets], -1)), -1)
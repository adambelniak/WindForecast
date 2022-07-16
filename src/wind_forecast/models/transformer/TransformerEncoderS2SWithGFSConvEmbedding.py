from typing import Dict

import torch
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.transformer.Transformer import PositionalEncoding, TransformerEncoderGFSBaseProps


class TransformerEncoderS2SWithGFSConvEmbedding(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.conv_embed_dim = config.experiment.conv_embedding_scale * self.features_length

        self.features_embeds = nn.Conv1d(in_channels=self.features_length, out_channels=self.conv_embed_dim,
                                         kernel_size=1)

        self.embed_dim = self.conv_embed_dim + self.dates_dim

        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.create_encoder()

        self.head_input_dim = self.embed_dim + 1
        self.create_head()

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

        input_elements = torch.cat(
            [*x, self.features_embeds((torch.cat(x, -1).permute(0, 2, 1))).permute(0, 2, 1)], -1)
        if self.config.experiment.with_dates_inputs:
            if self.use_time2vec:
                input_elements = torch.cat([input_elements, self.time_embed(dates_tensors[0])], -1)
            else:
                input_elements = torch.cat([input_elements, dates_tensors[0]], -1)

        x = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        memory = self.encoder(x)
        memory = memory[:, -self.future_sequence_length:, :]

        return torch.squeeze(self.classification_head(torch.cat([memory, gfs_targets], -1)), -1)

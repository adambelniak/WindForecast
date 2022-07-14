import math
from typing import Dict

import torch
import torch.nn as nn
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.transformer.Transformer import TransformerEncoderGFSBaseProps, PositionalEncoding
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2SCMAXWithGFS(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        out_channels = config.experiment.cnn_filters[-1]
        self.conv = CMAXEncoder(config)
        for _ in config.experiment.cnn_filters:
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)

        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv, config)
        self.conv_time_distributed = TimeDistributed(self.conv, batch_first=True)

        self.embed_dim += conv_W * conv_H * out_channels

        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        self.projection = TimeDistributed(nn.Linear(self.embed_dim, self.d_model), batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = self.prepare_elements_for_embedding(batch, False)
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        input_elements = torch.cat([input_elements, cmax_embeddings], -1)
        input_embedding = self.projection(input_elements)
        input_embedding = self.pos_encoder(input_embedding) if self.use_pos_encoding else input_embedding
        memory = self.encoder(input_embedding)
        memory = memory[:, -self.future_sequence_length:, :]

        return torch.squeeze(self.classification_head(torch.cat([self.forecaster(memory), gfs_targets], -1)), -1)

import math
from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.transformer.Transformer import PositionalEncoding, TransformerGFSBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerCMAXWithGFS(TransformerGFSBaseProps):
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
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.create_encoder()
        self.create_decoder()
        self.head_input_dim = self.embed_dim + 1
        self.create_head()

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, target_elements = self.prepare_elements_for_embedding(batch, is_train)

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        if is_train:
            cmax_targets = batch[BatchKeys.CMAX_FUTURE.value].float()

        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))
        if is_train:
            self.conv_time_distributed.requires_grad_(False)
            cmax_targets_embeddings = self.conv_time_distributed(cmax_targets.unsqueeze(2))
            self.conv_time_distributed.requires_grad_(True)

        input_elements = torch.cat([input_elements, cmax_embeddings], -1)
        if is_train:
            target_elements = torch.cat([target_elements, cmax_targets_embeddings], -1)

        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        if is_train:
            target_embedding = self.pos_encoder(target_elements) if self.use_pos_encoding else target_elements

        memory = self.encoder(input_embedding)
        output = self.base_transformer_forward(epoch, stage, input_embedding, target_embedding if is_train else None, memory)

        return torch.squeeze(self.classification_head(torch.cat([output, gfs_targets], -1)), -1)

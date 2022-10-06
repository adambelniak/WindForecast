import math
from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.tcn.HybridTCNS2S import HybridTCNS2S
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class HybridTCNS2SCMAX(HybridTCNS2S):
    def __init__(self, config: Config):
        super().__init__(config)
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        out_cnn_channels = config.experiment.cnn_filters[-1]
        self.conv_encoder = CMAXEncoder(config)
        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv_encoder, config)

        self.conv_time_distributed = TimeDistributed(self.conv_encoder, batch_first=True)

        for _ in config.experiment.cnn_filters:
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)

        self.embed_dim += conv_W * conv_H * out_cnn_channels
        self.create_tcn_encoder()
        self.create_tcn_decoder()
        self.regression_head_features = self.embed_dim
        if self.use_gfs and self.gfs_on_head:
            self.regression_head_features += 1

        self.create_regression_head()

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, all_gfs_targets = self.get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.use_gfs)

        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        cmax_embedding = self.conv_time_distributed(cmax_inputs.unsqueeze(2))
        input_elements = torch.cat([input_elements, cmax_embedding], dim=-1)

        x = self.encoder(input_elements.permute(0, 2, 1))[:, :, -self.future_sequence_length:]
        decoder_input = torch.cat([x, all_gfs_targets.permute(0, 2, 1)], -2)
        y = self.decoder(decoder_input).permute(0, 2, 1)[:, -self.future_sequence_length:, :]

        if self.use_gfs and self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return self.regressor_head(torch.cat([y, gfs_targets], -1)).squeeze(-1)
        return self.regressor_head(y).squeeze(-1)

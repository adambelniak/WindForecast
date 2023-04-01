import math
from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.transformer.Transformer import PositionalEncoding, TransformerEncoderBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed
from wind_forecast.util.common_util import get_pretrained_state_dict, get_pretrained_artifact_path


class TransformerEncoderS2SCMAXWithScaleToDepth(TransformerEncoderBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        self.scaling_factor = config.experiment.STD_scaling_factor
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
        self.head_input_dim = self.embed_dim
        self.pos_encoder = PositionalEncoding(self.embed_dim, self.dropout)
        self.create_encoder()
        self.create_head()

        if config.experiment.use_pretrained_artifact and type(self).__name__ is "TransformerEncoderS2SCMAXWithScaleToDepth":
            pretrained_autoencoder_path = get_pretrained_artifact_path(config.experiment.pretrained_artifact)
            self.load_state_dict(get_pretrained_state_dict(pretrained_autoencoder_path))
            return

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         False, False)

        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()
        cmax_inputs = cmax_inputs.unsqueeze(2)
        cmax_reshape = cmax_inputs.contiguous().view(cmax_inputs.size()[0] * cmax_inputs.size()[1], *cmax_inputs.size()[2:])
        cmax = torch.nn.functional.pixel_unshuffle(cmax_reshape, self.scaling_factor)
        cmax = cmax.contiguous().view(cmax_inputs.size(0), cmax_inputs.size(1), *cmax.size()[1:])

        cmax_embeddings = self.conv_time_distributed(cmax)

        input_elements = torch.cat([input_elements, cmax_embeddings], -1)
        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        memory = self.encoder(input_embedding)
        memory = memory[:, -self.future_sequence_length:, :]

        return torch.squeeze(self.regressor_head(memory), -1)

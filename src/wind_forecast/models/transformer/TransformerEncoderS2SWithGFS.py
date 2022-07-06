from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.transformer.Transformer import TransformerEncoderGFSBaseProps


class TransformerEncoderS2SWithGFS(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.future_sequence_length <= self.past_sequence_length, "Past sequence length can't be shorter than future sequence length for transformer encoder arch"

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = self.prepare_elements_for_embedding(batch, False)
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        input_embedding = self.projection(input_elements)
        input_embedding = self.pos_encoder(input_embedding) if self.use_pos_encoding else input_embedding
        memory = self.encoder(input_embedding)
        memory = memory[:, -self.future_sequence_length:, :]

        return torch.squeeze(self.classification_head_time_distributed(torch.cat([memory, gfs_targets], -1)), -1)

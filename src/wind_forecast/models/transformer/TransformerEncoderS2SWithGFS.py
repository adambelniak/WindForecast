from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.transformer.Transformer import TransformerEncoderGFSBaseProps


class TransformerEncoderS2SWithGFS(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        assert self.future_sequence_length <= self.past_sequence_length, "Past sequence length can't be shorter than future sequence length for transformer encoder arch"

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         True, False)
        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        memory = self.encoder(input_embedding)
        memory = memory[:, -self.future_sequence_length:, :]

        if self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return torch.squeeze(self.regressor_head(torch.cat([memory, gfs_targets], -1)), -1)

        return torch.squeeze(self.regressor_head(memory), -1)

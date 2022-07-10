from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.models.transformer.Transformer import TransformerEncoderBaseProps


class TransformerEncoderS2S(TransformerEncoderBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = self.prepare_elements_for_embedding(batch, False)
        input_embedding = self.projection(input_elements)
        input_embedding = self.pos_encoder(input_embedding) if self.use_pos_encoding else input_embedding
        memory = self.encoder(input_embedding)
        memory = memory[:, -self.future_sequence_length:, :]

        return torch.squeeze(self.classification_head(memory), -1)

from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.transformer.Transformer import TransformerEncoderBaseProps


class TransformerEncoder(TransformerEncoderBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         False, False)
        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        memory = self.encoder(input_embedding)
        memory = self.flatten(memory)
        return torch.squeeze(self.regressor_head(memory), -1)

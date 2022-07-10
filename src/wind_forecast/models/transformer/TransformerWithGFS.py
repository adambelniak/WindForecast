from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.transformer.Transformer import TransformerGFSBaseProps


class TransformerWithGFS(TransformerGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, target_elements = self.prepare_elements_for_embedding(batch, is_train)
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()

        input_embedding = self.projection(input_elements)
        input_embedding = self.pos_encoder(input_embedding) if self.use_pos_encoding else input_embedding
        if is_train:
            target_embedding = self.projection(target_elements)
            target_embedding = self.pos_encoder(target_embedding) if self.use_pos_encoding else target_embedding

        memory = self.encoder(input_embedding)
        output = self.base_transformer_forward(epoch, stage, input_embedding,
                                               target_embedding if is_train else None, memory)

        return torch.squeeze(self.classification_head(torch.cat([self.forecaster(output), gfs_targets], -1)), -1)

from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.transformer.Transformer import TransformerGFSBaseProps


class TransformerWithGFS(TransformerGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         True, is_train)
        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        if is_train:
            target_embedding = self.pos_encoder(target_elements) if self.use_pos_encoding else target_elements

        memory = self.encoder(input_embedding)
        output = self.base_transformer_forward(epoch, stage, input_embedding,
                                               target_embedding if is_train else None, memory)

        if self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return torch.squeeze(self.regressor_head(torch.cat([output, gfs_targets], -1)), -1)

        return torch.squeeze(self.regressor_head(output), -1)

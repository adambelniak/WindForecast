from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.transformer.Transformer import TransformerEncoderGFSBaseProps
from wind_forecast.util.common_util import get_pretrained_artifact_path, get_pretrained_state_dict


class Transformer(TransformerEncoderGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        if config.experiment.use_pretrained_artifact and type(self).__name__ is "Transformer":
            pretrained_autoencoder_path = get_pretrained_artifact_path(config.experiment.pretrained_artifact)
            self.load_state_dict(get_pretrained_state_dict(pretrained_autoencoder_path))
            return

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         True, False)
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()

        input_embedding = self.pos_encoder(input_elements) if self.use_pos_encoding else input_elements
        memory = self.encoder(input_embedding)
        memory = self.flatten(memory)

        return torch.squeeze(self.regressor_head(torch.cat([memory, gfs_targets], -1)), -1)


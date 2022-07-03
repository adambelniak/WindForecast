from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import TransformerEncoderBaseProps


class TransformerEncoder(TransformerEncoderBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        whole_input_embedding = torch.cat([synop_inputs, self.simple_2_vec_time_distributed(synop_inputs)], -1)
        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)
        memory = self.flatten(memory)
        return torch.squeeze(self.classification_head(memory), -1)

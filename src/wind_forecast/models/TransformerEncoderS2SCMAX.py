from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TransformerCMAX import TransformerCMAX


class TransformerEncoderS2SCMAX(TransformerCMAX):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_INPUTS.value].float()
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]
        cmax_inputs = batch[BatchKeys.CMAX_INPUTS.value].float()

        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        whole_input_embedding = torch.cat([synop_inputs, self.time_2_vec_time_distributed(synop_inputs), cmax_embeddings], -1)

        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        output = self.encoder(x)

        return torch.squeeze(self.classification_head_time_distributed(output), -1)

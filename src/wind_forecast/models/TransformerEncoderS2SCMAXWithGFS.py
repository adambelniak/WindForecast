from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.TransformerCMAXWithGFS import TransformerCMAXWithGFS


class TransformerEncoderS2SCMAXWithGFS(TransformerCMAXWithGFS):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        cmax_inputs = batch[BatchKeys.CMAX_PAST.value].float()

        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))

        if self.config.experiment.use_all_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            x = [synop_inputs, gfs_inputs]
        else:
            x = [synop_inputs]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1)), cmax_embeddings], -1)
        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        output = self.encoder(x)
        output = output[:, -self.future_sequence_length:, :]

        return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets], -1)), -1)


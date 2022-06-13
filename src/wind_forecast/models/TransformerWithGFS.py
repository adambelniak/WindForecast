from typing import Dict

import torch

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import TransformerGFSBaseProps


class TransformerWithGFS(TransformerGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        is_train = stage not in ['test', 'predict', 'validate']
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()

        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()

        if is_train:
            all_synop_targets = batch[BatchKeys.SYNOP_FUTURE_X.value].float()

        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.use_all_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float()
            x = [synop_inputs, gfs_inputs]
            if is_train:
                y = [all_synop_targets, all_gfs_targets]
        else:
            x = [synop_inputs]
            if is_train:
                y = [all_synop_targets]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)
        if is_train:
            whole_target_embedding = torch.cat([*y, self.time_2_vec_time_distributed(torch.cat(y, -1))], -1)

        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)
            if is_train:
                whole_target_embedding = torch.cat([whole_target_embedding, *dates_tensors[1]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        memory = self.encoder(x)
        output = self.base_transformer_forward(epoch, stage, whole_input_embedding, whole_target_embedding if is_train else None, memory)

        return torch.squeeze(self.classification_head_time_distributed(torch.cat([output, gfs_targets], -1)), -1)

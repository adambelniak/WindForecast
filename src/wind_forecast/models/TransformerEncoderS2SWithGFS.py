from typing import Dict

import torch
from torch import nn
from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import TransformerGFSBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2SWithGFS(TransformerGFSBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        dense_layers = []
        if self.config.experiment.with_dates_inputs:
            features = self.embed_dim + 7
        else:
            features = self.embed_dim + 1
        for neurons in self.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[
            BatchKeys.DATES_TENSORS.value]

        if self.config.experiment.use_all_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            x = [synop_inputs, gfs_inputs]
        else:
            x = [synop_inputs]

        whole_input_embedding = torch.cat([*x, self.time_2_vec_time_distributed(torch.cat(x, -1))], -1)

        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        x = self.encoder(x)
        x = x[:, -self.future_sequence_length:, :]

        if self.config.experiment.with_dates_inputs:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, gfs_targets, *dates_tensors[1]], -1)), -1)
        else:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, gfs_targets], -1)), -1)

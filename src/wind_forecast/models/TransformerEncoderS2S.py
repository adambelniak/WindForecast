from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.Transformer import TransformerEncoderBaseProps
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2S(TransformerEncoderBaseProps):
    def __init__(self, config: Config):
        super().__init__(config)
        dense_layers = []
        features = self.embed_dim
        if self.config.experiment.with_dates_inputs:
            features += 6
        for neurons in self.transformer_head_dims:
            dense_layers.append(nn.Linear(in_features=features, out_features=neurons))
            features = neurons
        dense_layers.append(nn.Linear(in_features=features, out_features=1))
        self.classification_head = nn.Sequential(*dense_layers)
        self.classification_head_time_distributed = TimeDistributed(self.classification_head, batch_first=True)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        if self.self_output_test:
            return self.self_forward(batch, epoch, stage)
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]

        whole_input_embedding = torch.cat([synop_inputs, self.simple_2_vec_time_distributed(synop_inputs)], -1)
        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        x = self.encoder(x)
        x = x[:, -self.future_sequence_length:, :]

        return torch.squeeze(self.classification_head_time_distributed(x), -1)

    def self_forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        synop_targets = batch[BatchKeys.SYNgOP_FUTURE_Y.value].float().unsqueeze(-1)
        dates_tensors = None if self.config.experiment.with_dates_inputs is False else batch[BatchKeys.DATES_TENSORS.value]
        whole_input_embedding = torch.cat([synop_targets, self.simple_2_vec_time_distributed(synop_targets)], -1)

        if self.config.experiment.with_dates_inputs:
            whole_input_embedding = torch.cat([whole_input_embedding, *dates_tensors[0]], -1)

        x = self.pos_encoder(whole_input_embedding) if self.use_pos_encoding else whole_input_embedding
        x = self.encoder(x)
        x = x[:, -self.future_sequence_length:, :]

        if self.config.experiment.with_dates_inputs:
            return torch.squeeze(self.classification_head_time_distributed(torch.cat([x, *dates_tensors[1]], -1)), -1)
        else:
            return torch.squeeze(self.classification_head_time_distributed(x), -1)
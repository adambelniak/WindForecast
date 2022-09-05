from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.models.tcn.TCNEncoder import TemporalBlock
from wind_forecast.models.tcn.TCNS2S import TCNS2S
from wind_forecast.models.value2vec.Value2Vec import Value2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class HybridTCNS2S(TCNS2S):
    def __init__(self, config: Config):
        super().__init__(config)
        assert config.experiment.use_gfs_data, "use_gfs_data needs to be true for hybrid model"

        self.gfs_embed_dim = self.gfs_params_len
        if self.use_value2vec:
            self.gfs_embed_dim += self.value2vec_embedding_factor * self.gfs_params_len
            self.value_embed_gfs = TimeDistributed(Value2Vec(self.gfs_params_len, self.value2vec_embedding_factor),
                                                   batch_first=True)
        self.create_tcn_decoder()

    def create_tcn_decoder(self):
        tcn_layers = []
        in_channels = self.config.experiment.tcn_channels[-1] + self.gfs_embed_dim
        for i in range(self.num_layers):
            dilation_size = 2 ** (self.num_layers - i)
            out_channels = self.tcn_channels[-(i + 2)] if i < self.num_layers - 1 else self.embed_dim
            tcn_layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, dilation=dilation_size,
                                         padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
            in_channels = out_channels

        self.decoder = nn.Sequential(*tcn_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, all_gfs_targets = self.get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.use_gfs)
        x = self.encoder(input_elements.permute(0, 2, 1))
        decoder_input = torch.cat([x, all_gfs_targets.permute(0, 2, 1)], -2)
        y = self.decoder(decoder_input).permute(0, 2, 1)[:, -self.future_sequence_length:, :]

        if self.use_gfs and self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return self.regressor_head(torch.cat([y, gfs_targets], -1)).squeeze(-1)
        return self.regressor_head(y).squeeze(-1)

    def get_embeddings(self, batch, with_dates, time_embed, with_gfs_params):
        synop_inputs = batch[BatchKeys.SYNOP_PAST_X.value].float()
        all_gfs_targets = batch[BatchKeys.GFS_FUTURE_X.value].float() if with_gfs_params else None
        dates_tensors = None if with_dates is False else batch[BatchKeys.DATES_TENSORS.value]

        if with_gfs_params:
            gfs_inputs = batch[BatchKeys.GFS_PAST_X.value].float()
            input_elements = torch.cat([synop_inputs, gfs_inputs], -1)
        else:
            input_elements = synop_inputs

        if self.use_value2vec:
            input_elements = torch.cat([input_elements, self.value_embed(input_elements)], -1)
            all_gfs_targets = torch.cat([all_gfs_targets, self.value_embed_gfs(all_gfs_targets)], -1)

        if with_dates:
            if time_embed is not None:
                input_elements = torch.cat([input_elements, time_embed(dates_tensors[0])], -1)
            else:
                input_elements = torch.cat([input_elements, dates_tensors[0]], -1)

        return input_elements, all_gfs_targets
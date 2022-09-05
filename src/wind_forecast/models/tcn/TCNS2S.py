from typing import Dict

import torch
import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.consts import BatchKeys
from wind_forecast.embed.prepare_embeddings import get_embeddings
from wind_forecast.models.tcn.TCNEncoder import TemporalBlock
from wind_forecast.models.tcn.TCNEncoderS2S import TCNEncoderS2S


class TCNS2S(TCNEncoderS2S):
    def __init__(self, config: Config):
        super().__init__(config)

        self.gfs_embed_dim = self.gfs_params_len
        self.create_tcn_decoder()
        self.regression_head_features = self.embed_dim
        if self.use_gfs and self.gfs_on_head:
            self.regression_head_features += 1

        self.create_regression_head()

    def create_tcn_decoder(self):
        tcn_layers = []
        in_channels = self.config.experiment.tcn_channels[-1]
        for i in range(self.num_layers):
            dilation_size = 2 ** (self.num_layers - i)
            out_channels = self.tcn_channels[-(i + 2)] if i < self.num_layers - 1 else self.embed_dim
            tcn_layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, dilation=dilation_size,
                                         padding=(self.kernel_size - 1) * dilation_size, dropout=self.dropout)]
            in_channels = out_channels

        self.decoder = nn.Sequential(*tcn_layers)

    def forward(self, batch: Dict[str, torch.Tensor], epoch: int, stage=None) -> torch.Tensor:
        input_elements, target_elements = get_embeddings(batch, self.config.experiment.with_dates_inputs,
                                                         self.time_embed if self.use_time2vec else None,
                                                         self.value_embed if self.use_value2vec else None,
                                                         self.use_gfs, False)
        x = self.encoder(input_elements.permute(0, 2, 1))
        y = self.decoder(x).permute(0, 2, 1)[:, -self.future_sequence_length:, :]

        if self.use_gfs and self.gfs_on_head:
            gfs_targets = batch[BatchKeys.GFS_FUTURE_Y.value].float()
            return self.regressor_head(torch.cat([y, gfs_targets], -1)).squeeze(-1)
        return self.regressor_head(y).squeeze(-1)

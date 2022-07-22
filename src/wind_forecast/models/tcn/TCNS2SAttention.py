import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.models.tcn.TCNEncoder import TemporalBlockWithAttention
from wind_forecast.models.tcn.TCNS2S import TCNS2S


class TCNS2SAttention(TCNS2S):
    def __init__(self, config: Config):
        super(TCNS2SAttention, self).__init__(config)

        tcn_layers = []
        in_channels = 1 if self.self_output_test or self.config.experiment.emd_decompose else self.embed_dim
        for i in range(self.num_levels):
            dilation_size = 2 ** i
            out_channels = self.tcn_channels[i]
            tcn_layers += [TemporalBlockWithAttention(config.experiment.transformer_attention_heads,
                                                      in_channels, out_channels, self.kernel_size,
                                                      dilation=dilation_size,
                                                      padding=(self.kernel_size - 1) * dilation_size,
                                                      dropout=self.dropout)]
            in_channels = self.tcn_channels[i]

        self.encoder = nn.Sequential(*tcn_layers)

        tcn_layers = []

        for i in range(self.num_levels):
            dilation_size = 2 ** (self.num_levels - i)
            out_channels = self.tcn_channels[-(i+2)] if i < self.num_levels - 1 else self.embed_dim
            tcn_layers += [TemporalBlockWithAttention(config.experiment.transformer_attention_heads,
                                                      in_channels, out_channels, self.kernel_size,
                                                      dilation=dilation_size,
                                                      padding=(self.kernel_size - 1) * dilation_size,
                                                      dropout=self.dropout)]
            in_channels = out_channels

        self.decoder = nn.Sequential(*tcn_layers)

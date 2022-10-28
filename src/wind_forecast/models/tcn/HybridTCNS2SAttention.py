import torch.nn as nn

from wind_forecast.config.register import Config
from wind_forecast.models.tcn.HybridTCNS2S import HybridTCNS2S
from wind_forecast.models.tcn.TCNEncoder import TemporalBlockWithAttention
from wind_forecast.util.common_util import get_pretrained_artifact_path, get_pretrained_state_dict


class HybridTCNS2SAttention(HybridTCNS2S):
    def __init__(self, config: Config):
        super().__init__(config)

        if config.experiment.use_pretrained_artifact and type(self).__name__ is "HybridTCNS2SAttention":
            pretrained_autoencoder_path = get_pretrained_artifact_path(config.experiment.pretrained_artifact)
            self.load_state_dict(get_pretrained_state_dict(pretrained_autoencoder_path))
            return

    def create_tcn_encoder(self):
        tcn_layers = []
        in_channels = 1 if self.self_output_test or self.config.experiment.emd_decompose else self.embed_dim
        for i in range(self.num_layers):
            dilation_size = 2 ** i
            out_channels = self.tcn_channels[i]
            tcn_layers += [TemporalBlockWithAttention(self.config.experiment.transformer_attention_heads,
                                                      in_channels, out_channels, self.kernel_size,
                                                      dilation=dilation_size,
                                                      padding=(self.kernel_size - 1) * dilation_size,
                                                      dropout=self.dropout)]
            in_channels = self.tcn_channels[i]

        self.encoder = nn.Sequential(*tcn_layers)

    def create_tcn_decoder(self):
        tcn_layers = []
        in_channels = self.config.experiment.tcn_channels[-1] + self.gfs_embed_dim

        for i in range(self.num_layers):
            dilation_size = 2 ** (self.num_layers - i)
            out_channels = self.tcn_channels[-(i + 2)] if i < self.num_layers - 1 else self.embed_dim
            tcn_layers += [TemporalBlockWithAttention(self.config.experiment.transformer_attention_heads,
                                                      in_channels, out_channels, self.kernel_size,
                                                      dilation=dilation_size,
                                                      padding=(self.kernel_size - 1) * dilation_size,
                                                      dropout=self.dropout)]
            in_channels = out_channels

        self.decoder = nn.Sequential(*tcn_layers)

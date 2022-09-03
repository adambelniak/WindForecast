import math

from wind_forecast.config.register import Config
from wind_forecast.models.CMAXAutoencoder import CMAXEncoder, get_pretrained_encoder
from wind_forecast.models.tcn.TCNS2SAttention import TCNS2SAttention
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TCNS2SAttentionCMAX(TCNS2SAttention):
    def __init__(self, config: Config):
        super().__init__(config)
        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        out_cnn_channels = config.experiment.cnn_filters[-1]
        self.conv_encoder = CMAXEncoder(config)
        if config.experiment.use_pretrained_cmax_autoencoder:
            get_pretrained_encoder(self.conv_encoder, config)

        self.conv_time_distributed = TimeDistributed(self.conv_encoder, batch_first=True)

        for _ in config.experiment.cnn_filters:
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)

        self.embed_dim += conv_W * conv_H * out_cnn_channels
        self.create_tcn_encoder()
        self.create_tcn_decoder()
        self.regression_head_features = self.embed_dim
        if self.use_gfs and self.gfs_on_head:
            self.regression_head_features += 1

        self.create_regression_head()
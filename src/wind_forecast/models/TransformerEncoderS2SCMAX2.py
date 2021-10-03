import torch
from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config
from wind_forecast.models.Transformer import PositionalEncoding
from wind_forecast.models.TransformerEncoder import Time2Vec
from wind_forecast.time_distributed.TimeDistributed import TimeDistributed


class TransformerEncoderS2SCMAX2(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        features_len = len(config.experiment.synop_train_features)
        embed_dim = (config.experiment.time2vec_embedding_size + 1) * features_len + 256  # 256 is the output size from conv
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features=16),
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features=32),
                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features=32),
                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features=32),
                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features=32),
                                  nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(num_features=16),
                                  nn.Flatten()
                                  )

        self.conv_time_distributed = TimeDistributed(self.conv)

        self.time2vec = Time2Vec(config)
        self.pos_encoder = PositionalEncoding(embed_dim, config.experiment.dropout, config.experiment.sequence_length)
        self.time2vec_time_distributed = TimeDistributed(self.time2vec, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=config.experiment.transformer_attention_heads,
                                                   dim_feedforward=config.experiment.transformer_ff_dim, dropout=config.experiment.dropout,
                                                   batch_first=True)
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.experiment.transformer_attention_layers, encoder_norm)
        self.linear = nn.Linear(in_features=embed_dim, out_features=1)
        self.linear_time_distributed = TimeDistributed(self.linear, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, inputs, cmax_inputs, targets: torch.Tensor, epoch: int, stage=None):
        cmax_embeddings = self.conv_time_distributed(cmax_inputs.unsqueeze(2))
        inputs_embeddings = self.time2vec_time_distributed(inputs)
        x = torch.cat([inputs, inputs_embeddings, cmax_embeddings], -1)
        x = self.pos_encoder(x)
        x = self.encoder(x)

        return torch.squeeze(self.linear_time_distributed(x), dim=-1)


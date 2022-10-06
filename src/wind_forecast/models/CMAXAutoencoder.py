from pytorch_lightning import LightningModule
from torch import nn
import torch
from torch.nn import Parameter

from wind_forecast.config.register import Config
from wind_forecast.util.common_util import get_pretrained_artifact_path
import math


def get_pretrained_encoder(module: nn.Module, config: Config):
    pretrained_autoencoder_path = get_pretrained_artifact_path(config.experiment.pretrained_artifact)
    checkpoint = torch.load(pretrained_autoencoder_path)
    state_dict = checkpoint['state_dict']
    state_dict = {k.partition('model.')[2]:state_dict[k] for k in state_dict.keys()}
    pretrained_model = CMAXAutoencoder(config=config)
    pretrained_model.load_state_dict(state_dict)
    module_state = module.state_dict()
    pretrained_state_dict = pretrained_model.state_dict()
    for name, param in pretrained_state_dict.items():
        if name not in module_state or not name.startswith('encoder'):
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        module_state[name].copy_(param)


class CMAXEncoder(LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        conv_H = config.experiment.cmax_h
        conv_W = config.experiment.cmax_w
        conv_layers = []
        assert len(config.experiment.cnn_filters) > 0

        in_channels = 1
        for index, filters in enumerate(config.experiment.cnn_filters):
            out_channels = filters
            conv_layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2),
                          padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=out_channels),
            ])
            if index != len(config.experiment.cnn_filters) - 1:
                conv_layers.append(nn.Dropout(config.experiment.dropout))
            conv_W = math.ceil(conv_W / 2)
            conv_H = math.ceil(conv_H / 2)
            in_channels = out_channels

        self.encoder = nn.Sequential(*conv_layers,
                                  nn.Flatten(),
                                  nn.Linear(in_features=conv_W * conv_H * out_channels,
                                            out_features=conv_W * conv_H * out_channels))

        self.output_shape = (conv_W * conv_H * out_channels, )

    def forward(self, inputs: torch.Tensor):
        return self.encoder(inputs)

    def get_output_shape(self):
        return self.output_shape


class CMAXDecoder(LightningModule):
    def __init__(self, config: Config, input_shape):
        super().__init__()
        in_channels = config.experiment.cnn_filters[-1]
        in_size = int(math.sqrt(input_shape[0] / in_channels))
        conv_layers = []
        filters_layers = config.experiment.cnn_filters[:-1]
        filters_layers.reverse()

        for index, filters in enumerate(filters_layers):
            out_channels = filters
            conv_layers.extend([
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(2, 2),
                          padding=1, output_padding= 1 if index == 0 else 0),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=out_channels),
            ])
            if index != len(filters_layers) - 1:
                conv_layers.append(nn.Dropout(config.experiment.dropout))
            in_channels = out_channels

        conv_layers.extend([
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=1, kernel_size=(3, 3), stride=(2, 2),
                               padding=1, output_padding=0)
        ])

        self.decoder = nn.Sequential(nn.Unflatten(dim=1, unflattened_size=(in_channels, in_size, in_size)),
                                     nn.BatchNorm2d(num_features=in_channels),
                                     *conv_layers)

    def forward(self, inputs: torch.Tensor):
        return self.decoder(inputs)


class CMAXAutoencoder(LightningModule):
    def __init__(self, config: Config):
        super().__init__()

        self.encoder = CMAXEncoder(config)
        self.decoder = CMAXDecoder(config, input_shape=self.encoder.get_output_shape())

    def forward(self, cmax_inputs):
        if len(cmax_inputs.size()) == 3:
            # 1 channel
            cmax_inputs = cmax_inputs.unsqueeze(1)
        hidden = self.encoder(cmax_inputs)
        if len(cmax_inputs.size()) == 3:
            return self.decoder(hidden)

        return self.decoder(hidden).squeeze()



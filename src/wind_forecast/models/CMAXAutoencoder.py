from pytorch_lightning import LightningModule
from torch import nn

from wind_forecast.config.register import Config


class CMAXAutoencoder(LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=1),
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
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.Flatten(),
                                     nn.Linear(in_features=512, out_features=512),
                                     nn.Unflatten(dim=1, unflattened_size=(32, 4, 4)),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=1,
                                                        output_padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                                                        stride=(2, 2), padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                                                        stride=(2, 2), padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                                                        stride=(2, 2), padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3),
                                                        stride=(2, 2), padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=16),
                                     nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 3),
                                                        stride=(2, 2), padding=1)
                                     )

    def forward(self, cmax_inputs):
        return self.network(cmax_inputs)



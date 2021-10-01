from pytorch_lightning import LightningModule
from torch import nn


class CMAXAutoencoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=16),
                                     # nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1)),
                                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     # nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1)),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     # nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1)),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     # nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1)),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     # nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1)),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.Flatten(),
                                     nn.Linear(in_features=128, out_features=128),
                                     nn.Unflatten(dim=128, unflattened_size=(32, 2, 2)),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                                                        stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                                                        stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3),
                                                        stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=32),
                                     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3),
                                                        stride=(2, 2)),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(num_features=16),
                                     nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 3),
                                                        stride=(2, 2))
                                     )

    def forward(self, cmax_inputs):
        return self.network(cmax_inputs)



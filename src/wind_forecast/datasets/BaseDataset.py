import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.mean = ...
        self.std = ...

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std
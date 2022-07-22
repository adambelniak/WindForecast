import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.mean = self.std = self.min = self.max = self.gfs_mean = self.gfs_std = None

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def set_gfs_mean(self, mean):
        self.gfs_mean = mean

    def set_gfs_std(self, std):
        self.gfs_std = std

    def set_min(self, min):
        self.min = min

    def set_max(self, max):
        self.max = max
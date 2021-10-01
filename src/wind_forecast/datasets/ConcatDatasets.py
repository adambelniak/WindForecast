import torch


class ConcatDatasets(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.mean = [dataset.mean for dataset in datasets]
        self.std = [dataset.std for dataset in datasets]

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
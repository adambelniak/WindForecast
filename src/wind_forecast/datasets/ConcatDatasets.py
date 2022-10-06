from wind_forecast.datasets.BaseDataset import BaseDataset


class ConcatDatasets(BaseDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def set_mean(self, mean):
        assert len(mean) == len(self.datasets)
        for index, dataset in enumerate(self.datasets):
            dataset.set_mean(mean[index])
        self.mean = mean

    def set_std(self, std):
        assert len(std) == len(self.datasets)
        for index, dataset in enumerate(self.datasets):
            dataset.set_std(std[index])
        self.std = std

    def set_min(self, min):
        assert len(min) == len(self.datasets)
        for index, dataset in enumerate(self.datasets):
            dataset.set_min(min[index])
        self.min = min

    def set_max(self, max):
        assert len(max) == len(self.datasets)
        for index, dataset in enumerate(self.datasets):
            dataset.set_max(max[index])
        self.max = max

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

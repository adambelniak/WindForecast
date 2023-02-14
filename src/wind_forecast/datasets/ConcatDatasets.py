from wind_forecast.datasets.BaseDataset import BaseDataset


class ConcatDatasets(BaseDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.SYNOP_PAST_Y_INDEX = datasets[0].SYNOP_PAST_Y_INDEX
        self.SYNOP_PAST_X_INDEX = datasets[0].SYNOP_PAST_X_INDEX
        self.SYNOP_FUTURE_Y_INDEX = datasets[0].SYNOP_FUTURE_Y_INDEX
        self.SYNOP_FUTURE_X_INDEX = datasets[0].SYNOP_FUTURE_X_INDEX
        self.GFS_PAST_X_INDEX = datasets[0].GFS_PAST_X_INDEX
        self.GFS_PAST_Y_INDEX = datasets[0].GFS_PAST_Y_INDEX
        self.GFS_FUTURE_X_INDEX = datasets[0].GFS_FUTURE_X_INDEX
        self.GFS_FUTURE_Y_INDEX = datasets[0].GFS_FUTURE_Y_INDEX
        self.DATES_PAST_INDEX = datasets[0].DATES_PAST_INDEX
        self.DATES_FUTURE_INDEX = datasets[0].DATES_FUTURE_INDEX

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

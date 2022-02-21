from wind_forecast.loaders.Singleton import Singleton


class DataModulesCache(metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.datasets = {}

    def cache_dataset(self, dataset_name: str, dataset):
        self.datasets[dataset_name] = dataset

    def get_cached_dataset(self, dataset_name: str):
        if dataset_name in self.datasets.keys():
            return self.datasets[dataset_name]
        return None

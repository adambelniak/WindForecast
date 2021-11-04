from wind_forecast.loaders.Singleton import Singleton


class DataModulesCache(metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = None

    def cache_dataset(self, dataset):
        self.dataset = dataset

    def get_cached_dataset(self):
        return self.dataset

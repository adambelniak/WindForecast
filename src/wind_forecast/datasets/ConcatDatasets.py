from wind_forecast.datasets.BaseDataset import BaseDataset


class ConcatDatasets(BaseDataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def get_dataset(self, class_name):
        for dataset in self.datasets:
            if type(dataset).__name__ == class_name:
                return dataset
        raise Exception(f"Dataset of class {class_name} not found.")

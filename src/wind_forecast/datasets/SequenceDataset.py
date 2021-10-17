from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset


class SequenceDataset(BaseDataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, synop_data, synop_data_indices):
        'Initialization'
        super().__init__()
        self.target_param = config.experiment.target_parameter
        train_params = config.experiment.synop_train_features
        self.sequence_length = config.experiment.sequence_length
        self.synop_data = synop_data
        self.prediction_offset = config.experiment.prediction_offset
        self.train_params = list(list(zip(*train_params))[1])
        self.data = synop_data_indices

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        synop_index = self.data[index]
        x = self.synop_data.iloc[synop_index:synop_index + self.sequence_length][self.train_params].to_numpy()
        y = self.synop_data.iloc[synop_index + self.sequence_length + self.prediction_offset][self.target_param]

        return x, y

import torch
import numpy as np
from tqdm import tqdm

from wind_forecast.config.register import Config


class SequenceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, synop_data, dates, train=True):
        'Initialization'
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data.reset_index()
        synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index

        self.features = [self.synop_data.iloc[index:index + self.sequence_length][list(list(zip(*self.train_params))[1])].to_numpy()
                         for index in tqdm(synop_data_indices)]

        self.targets = [self.synop_data.iloc[index + self.sequence_length + self.prediction_offset][self.target_param]
                        for index in tqdm(synop_data_indices)]

        assert len(self.features) == len(self.targets)
        length = len(self.targets)
        training_data = np.array(list(zip(self.features, self.targets))[:int(length * 0.8)])
        test_data = np.array(list(zip(self.features, self.targets))[int(length * 0.8) + self.sequence_length - 1:])
        if train:
            data = training_data
        else:
            data = test_data

        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y = self.data[index][0], self.data[index][1]

        return x, y

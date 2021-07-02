import torch
import numpy as np

from wind_forecast.config.register import Config
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset


class SequenceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, train=True):
        'Initialization'
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.lstm_train_parameters
        self.synop_file = config.experiment.synop_file
        raw_data = prepare_synop_dataset(self.synop_file, list(list(zip(*self.train_params))[1]))
        labels = raw_data[self.target_param].to_numpy()
        features = raw_data[list(list(zip(*self.train_params))[1])].to_numpy()
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset

        self.features = [features[i:i + self.sequence_length, :].T for i in
                         range(features.shape[0] - self.sequence_length - self.prediction_offset + 1)]
        self.targets = [labels[i + self.sequence_length + self.prediction_offset - 1].T for i in
                        range(labels.shape[0] - self.sequence_length - self.prediction_offset + 1)]
        self.features = np.array(self.features).reshape((len(self.features), self.sequence_length, len(self.train_params)))
        assert len(self.features) == len(self.targets)
        length = len(self.targets)
        training_data = np.array(list(zip(self.features, self.targets))[:int(length * 0.8)])
        test_data = np.array(list(zip(self.features, self.targets))[int(length * 0.8):])
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

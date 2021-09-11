import torch
import numpy as np

from wind_forecast.config.register import Config


class SequenceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, labels, train=True):
        'Initialization'
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.synop_file = config.experiment.synop_file
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.labels = labels

        labels = self.labels[self.target_param].to_numpy()
        features = self.labels[list(list(zip(*self.train_params))[1])].to_numpy()

        self.features = [features[i:i + self.sequence_length, :].T for i in
                         range(features.shape[0] - self.sequence_length - self.prediction_offset + 1)]
        self.targets = [labels[i + self.sequence_length + self.prediction_offset - 1].T for i in
                        range(labels.shape[0] - self.sequence_length - self.prediction_offset + 1)]
        self.features = np.transpose(np.array(self.features), (0, 2, 1))
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

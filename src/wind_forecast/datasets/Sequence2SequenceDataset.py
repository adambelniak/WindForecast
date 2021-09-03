import torch
import numpy as np

from wind_forecast.config.register import Config
from wind_forecast.consts import SYNOP_DATASETS_DIRECTORY
from wind_forecast.preprocess.synop.synop_preprocess import prepare_synop_dataset


class Sequence2SequenceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, train=True):
        'Initialization'
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.synop_file = config.experiment.synop_file
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset

        raw_data, synop_feature_1, synop_feature_2 = prepare_synop_dataset(self.synop_file,
                                                                           list(list(zip(*self.train_params))[1]),
                                                                           dataset_dir=SYNOP_DATASETS_DIRECTORY)
        features = raw_data[list(list(zip(*self.train_params))[1])].to_numpy()
        labels = raw_data[self.target_param].to_numpy()

        self.features = [features[i:i + self.sequence_length, :].T for i in
                         range(features.shape[0] - self.sequence_length - self.prediction_offset - self.future_sequence_length + 1)]
        self.all_targets = [features[
                        i + self.sequence_length + self.prediction_offset - 1: i + self.sequence_length + self.prediction_offset - 1 + self.future_sequence_length]
                            for i in
                            range(features.shape[0] - self.sequence_length - self.prediction_offset + 1 - self.future_sequence_length)]
        self.targets = [labels[
                            i + self.sequence_length + self.prediction_offset - 1: i + self.sequence_length + self.prediction_offset - 1 + self.future_sequence_length]
                            for i in
                            range(features.shape[
                                      0] - self.sequence_length - self.prediction_offset + 1 - self.future_sequence_length)]

        self.features = np.transpose(np.array(self.features), (0, 2, 1))

        assert len(self.features) == len(self.all_targets)
        assert len(self.features) == len(self.targets)

        length = len(self.all_targets)
        training_data = list(zip(zip(self.features, self.all_targets), self.targets))[:int(length * 0.8)]
        test_data = list(zip(zip(self.features, self.all_targets), self.targets))[int(length * 0.8):]
        if train:
            data = training_data
        else:
            data = test_data

        self.data = data
        target_param_index = [x[1] for x in self.train_params].index(self.target_param)
        print(synop_feature_1[target_param_index])
        print(synop_feature_2[target_param_index])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y = self.data[index][0], self.data[index][1]
        inputs, all_targets = x[0], x[1]
        return inputs, all_targets, y

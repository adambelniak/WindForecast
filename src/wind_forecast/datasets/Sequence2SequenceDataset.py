import torch
from tqdm import tqdm
from wind_forecast.config.register import Config


class Sequence2SequenceDataset(torch.utils.data.Dataset):
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
        target_param_index = [x[1] for x in self.train_params].index(self.target_param)

        self.features = [self.synop_data.iloc[index:index + self.sequence_length][list(list(zip(*self.train_params))[1])].to_numpy()
                                for index in tqdm(synop_data_indices)]

        self.all_targets = [self.synop_data.iloc[
                            index + self.sequence_length + self.prediction_offset:index + self.sequence_length + self.prediction_offset + self.future_sequence_length][
                                list(list(zip(*self.train_params))[1])].to_numpy() for index in tqdm(synop_data_indices)]

        self.targets = [target[:,target_param_index] for target in self.all_targets]

        assert len(self.features) == len(self.all_targets)
        assert len(self.features) == len(self.targets)

        length = len(self.features)
        training_data = list(zip(zip(self.features, self.all_targets), self.targets))[:int(length * 0.8)]
        # do not use any frame from train set in test set
        test_data = list(zip(zip(self.features, self.all_targets), self.targets))[int(length * 0.8) + self.sequence_length - 1:]

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
        inputs, all_targets = x[0], x[1]
        return inputs, all_targets, y

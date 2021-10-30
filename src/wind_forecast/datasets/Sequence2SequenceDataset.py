from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset
from wind_forecast.util.gfs_util import add_param_to_train_params


class Sequence2SequenceDataset(BaseDataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, synop_data, synop_data_indices):
        super().__init__()
        'Initialization'
        self.target_param = config.experiment.target_parameter
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data.reset_index()
        train_params = config.experiment.synop_train_features
        self.target_param = config.experiment.target_parameter
        all_params = add_param_to_train_params(train_params, self.target_param)
        self.feature_names = list(list(zip(*all_params))[1])
        self.target_param_index = [x for x in self.feature_names].index(self.target_param)
        self.train_params = list(list(zip(*train_params))[1])
        self.data = synop_data_indices

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        synop_index = self.data[index]
        inputs = self.synop_data.iloc[synop_index:synop_index + self.sequence_length][self.train_params].to_numpy()
        all_targets = self.synop_data.iloc[
                      synop_index + self.sequence_length + self.prediction_offset:synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length][
                                self.train_params].to_numpy()
        y = all_targets[:, self.target_param_index]
        inputs_dates = self.synop_data.iloc[synop_index:synop_index + self.sequence_length]['date'].to_numpy()
        y_dates = self.synop_data.iloc[synop_index + self.sequence_length + self.prediction_offset:synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length]['date']
        return inputs, all_targets, y, y_dates, inputs_dates

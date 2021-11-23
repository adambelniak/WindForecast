from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset
from wind_forecast.util.gfs_util import add_param_to_train_params


class Sequence2SequenceWithGFSDataset(BaseDataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, synop_data, synop_data_indices, gfs_target_data, all_gfs_target_data=None, all_gfs_input_data=None):
        'Initialization'
        super().__init__()
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
        self.use_all_gfs_params = config.experiment.use_all_gfs_params

        if self.use_all_gfs_params:
            self.data = list(zip(synop_data_indices, all_gfs_input_data, gfs_target_data, all_gfs_target_data))
        else:
            self.data = list(zip(synop_data_indices, gfs_target_data))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.use_all_gfs_params:
            synop_index, all_gfs_inputs, gfs_targets, all_gfs_target_data = self.data[index]
        else:
            synop_index, gfs_targets = self.data[index]

        synop_inputs = self.synop_data.iloc[synop_index:synop_index + self.sequence_length][self.train_params].to_numpy()
        all_synop_targets = self.synop_data.iloc[
                      synop_index + self.sequence_length + self.prediction_offset:synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length][
            self.train_params].to_numpy()
        synop_targets = all_synop_targets[:, self.target_param_index]
        inputs_dates = self.synop_data.iloc[synop_index:synop_index + self.sequence_length]['date']
        target_dates = self.synop_data.iloc[
                  synop_index + self.sequence_length + self.prediction_offset
                  :synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length]['date']

        if self.use_all_gfs_params:
            return synop_inputs, synop_targets, all_synop_targets, all_gfs_inputs, gfs_targets, all_gfs_target_data, inputs_dates, target_dates

        return synop_inputs, synop_targets, all_synop_targets, gfs_targets, inputs_dates, target_dates

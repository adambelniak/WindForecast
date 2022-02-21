from typing import List

from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset


class Sequence2SequenceWithGFSDataset(BaseDataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, synop_data, synop_data_indices, synop_feature_names: List[str], gfs_future_y, gfs_future_x=None, gfs_past_x=None):
        'Initialization'
        super().__init__()
        self.train_params = synop_feature_names
        self.target_param = config.experiment.target_parameter
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data.reset_index()
        self.use_all_gfs_params = config.experiment.use_all_gfs_params

        if self.use_all_gfs_params:
            self.data = list(zip(synop_data_indices, gfs_past_x, gfs_future_y, gfs_future_x))
        else:
            self.data = list(zip(synop_data_indices, gfs_future_y))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.use_all_gfs_params:
            synop_index, gfs_past_x, gfs_future_y, gfs_future_x = self.data[index]
        else:
            synop_index, gfs_future_y = self.data[index]

        synop_past_x = self.synop_data.iloc[synop_index:synop_index + self.sequence_length][self.train_params].to_numpy()
        synop_future_x = self.synop_data.iloc[
                      synop_index + self.sequence_length + self.prediction_offset:synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length][
            self.train_params].to_numpy()
        synop_y = self.synop_data.iloc[
                      synop_index:synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length][
            self.target_param].to_numpy()
        synop_past_y = synop_y[:self.sequence_length + self.prediction_offset]
        synop_future_y = synop_y[self.sequence_length + self.prediction_offset
                                             :synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length]
        inputs_dates = self.synop_data.iloc[synop_index:synop_index + self.sequence_length]['date']
        target_dates = self.synop_data.iloc[
                  synop_index + self.sequence_length + self.prediction_offset
                  :synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length]['date']

        if self.use_all_gfs_params:
            return synop_past_y, synop_past_x, synop_future_y, synop_future_x, gfs_past_x,\
                   gfs_future_y, gfs_future_x, inputs_dates, target_dates

        return synop_past_y, synop_past_x, synop_future_y, synop_future_x, gfs_future_y, inputs_dates, target_dates

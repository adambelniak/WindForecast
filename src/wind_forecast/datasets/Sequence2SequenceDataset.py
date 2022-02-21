from typing import List

from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset


class Sequence2SequenceDataset(BaseDataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, synop_data, synop_data_indices, synop_feature_names: List[str]):
        super().__init__()
        'Initialization'
        self.train_params = synop_feature_names
        self.target_param = config.experiment.target_parameter
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data.reset_index()
        self.data = synop_data_indices

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        synop_index = self.data[index]
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
        past_dates = self.synop_data.iloc[synop_index:synop_index + self.sequence_length]['date']
        future_dates = self.synop_data.iloc[synop_index + self.sequence_length + self.prediction_offset:synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length]['date']

        return synop_past_y, synop_past_x, synop_future_y, synop_future_x, past_dates, future_dates

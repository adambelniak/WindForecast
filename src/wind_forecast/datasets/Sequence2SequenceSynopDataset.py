from typing import List

import pandas as pd

from wind_forecast.config.register import Config
from wind_forecast.datasets.BaseDataset import BaseDataset


class Sequence2SequenceSynopDataset(BaseDataset):

    def __init__(self, config: Config, synop_data: pd.DataFrame, data_indices: list, synop_feature_names: List[str]):
        'Initialization'
        super().__init__()
        self.synop_feature_names = synop_feature_names
        self.target_param = config.experiment.target_parameter

        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data

        self.data = data_indices

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        data_index = self.data[index]

        synop_past_x = self.synop_data.loc[data_index:data_index + self.sequence_length - 1][
            self.synop_feature_names].to_numpy()
        synop_future_x = self.synop_data.loc[
                         data_index + self.sequence_length + self.prediction_offset:data_index + self.sequence_length + self.prediction_offset + self.future_sequence_length - 1][
            self.synop_feature_names].to_numpy()

        synop_y = self.synop_data.loc[
                  data_index:data_index + self.sequence_length + self.prediction_offset + self.future_sequence_length - 1][
            self.target_param].to_numpy()
        synop_past_y = synop_y[:self.sequence_length]
        synop_future_y = synop_y[self.sequence_length + self.prediction_offset
                                 :self.sequence_length + self.prediction_offset + self.future_sequence_length]

        inputs_dates = self.synop_data.loc[data_index:data_index + self.sequence_length - 1]['date'].to_numpy()
        target_dates = self.synop_data.loc[
                       data_index + self.sequence_length + self.prediction_offset
                       :data_index + self.sequence_length + self.prediction_offset + self.future_sequence_length - 1]['date'].to_numpy()

        return synop_past_y, synop_past_x, synop_future_y, synop_future_x, inputs_dates, target_dates


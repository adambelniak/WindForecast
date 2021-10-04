import torch

from wind_forecast.config.register import Config
from wind_forecast.preprocess.synop.synop_preprocess import normalize_synop_data
from wind_forecast.util.gfs_util import add_param_to_train_params


class Sequence2SequenceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, config: Config, synop_data, dates, normalize_synop=True):
        'Initialization'
        self.target_param = config.experiment.target_parameter
        train_params = config.experiment.synop_train_features
        self.sequence_length = config.experiment.sequence_length
        self.future_sequence_length = config.experiment.future_sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data.reset_index()
        self.mean = ...
        self.std = ...
        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index
        params = add_param_to_train_params(train_params, self.target_param)
        feature_names = list(list(zip(*params))[1])
        self.target_param_index = [x for x in feature_names].index(self.target_param)
        if normalize_synop:
            # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
            self.synop_data, synop_mean, synop_std = normalize_synop_data(self.synop_data, synop_data_indices,
                                                                          feature_names,
                                                                          self.sequence_length + self.prediction_offset
                                                                          + self.future_sequence_length,
                                                                          config.experiment.normalization_type)
            self.mean = synop_mean[self.target_param_index]
            self.std = synop_std[self.target_param_index]
            # if data was already normalized we don't know the mean and std, but it's YANGNI now
            print(self.mean)
            print(self.std)

        self.train_params = list(list(zip(*train_params))[1])

        print(len(synop_data_indices))
        self.data = synop_data_indices

    def __len__(self):
        'Denotes the total number of samples'
        print(len(self.data))
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        synop_index = self.data[index]
        inputs = self.synop_data.iloc[synop_index:synop_index + self.sequence_length][self.train_params].to_numpy()
        all_targets = self.synop_data.iloc[
                      synop_index + self.sequence_length + self.prediction_offset:synop_index + self.sequence_length + self.prediction_offset + self.future_sequence_length][
                                self.train_params].to_numpy()
        y = all_targets[:, self.target_param_index]
        return inputs, all_targets, y

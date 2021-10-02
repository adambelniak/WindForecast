import numpy as np
import torch
from tqdm import tqdm

from wind_forecast.config.register import Config
from wind_forecast.preprocess.synop.synop_preprocess import normalize_synop_data
from wind_forecast.util.gfs_util import add_param_to_train_params


class SequenceDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, config: Config, synop_data, dates, normalize_synop=True):
        'Initialization'
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.synop_data = synop_data.reset_index()
        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index
        params = add_param_to_train_params(self.train_params, self.target_param)
        feature_names = list(list(zip(*params))[1])
        if normalize_synop:
            # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
            self.synop_data, synop_mean, synop_std = normalize_synop_data(self.synop_data, synop_data_indices,
                                                                          feature_names,
                                                                          self.sequence_length + self.prediction_offset,
                                                                          config.experiment.normalization_type)
            target_param_index = [x for x in feature_names].index(self.target_param)
            print(synop_mean[target_param_index])
            print(synop_std[target_param_index])

        self.features = []
        self.targets = []

        print("Preparing the dataset")
        for index in tqdm(synop_data_indices):
            self.features.append(self.synop_data.iloc[index:index + self.sequence_length][list(list(zip(*self.train_params))[1])].to_numpy())
            self.targets.append(self.synop_data.iloc[index + self.sequence_length + self.prediction_offset][self.target_param])

        assert len(self.features) == len(self.targets)

        self.data = np.array(list(zip(self.features, self.targets)))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y = self.data[index][0], self.data[index][1]

        return x, y

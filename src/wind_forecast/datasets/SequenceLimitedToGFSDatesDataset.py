import pandas as pd
import torch
from tqdm import tqdm

from wind_forecast.config.register import Config
from wind_forecast.preprocess.synop.synop_preprocess import normalize_synop_data_for_training
from wind_forecast.util.gfs_util import add_param_to_train_params, match_gfs_with_synop_sequence, \
    target_param_to_gfs_name_level


class SequenceLimitedToGFSDatesDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, config: Config, synop_data, dates, normalize_synop=True):
        """Initialization"""
        self.target_param = config.experiment.target_parameter
        self.train_params = config.experiment.synop_train_features
        self.synop_file = config.experiment.synop_file
        self.sequence_length = config.experiment.sequence_length
        self.prediction_offset = config.experiment.prediction_offset
        self.target_coords = config.experiment.target_coords

        self.synop_data = synop_data.reset_index()
        # Get indices which correspond to 'dates' - 'dates' are the ones, which start a proper sequence without breaks
        synop_data_indices = self.synop_data[self.synop_data["date"].isin(dates)].index

        all_params = add_param_to_train_params(self.train_params, self.target_param)
        feature_names = list(list(zip(*all_params))[1])

        if normalize_synop:
            # data was not normalized, so take all frames which will be used, compute std and mean and normalize data
            self.synop_data, self.synop_feature_names, synop_feature_1, synop_feature_2 = \
                normalize_synop_data_for_training(self.synop_data,
                                                  synop_data_indices,
                                                  feature_names,
                                                  self.sequence_length + self.prediction_offset,
                                                  self.target_param,
                                                  config.experiment.normalization_type)
            target_param_index = [x for x in feature_names].index(self.target_param)
            print(synop_feature_1[target_param_index])
            print(synop_feature_2[target_param_index])

        synop_data_dates = self.synop_data['date']
        # labels and dates - dates are needed for matching the labels against GFS dates
        labels = pd.concat([synop_data_dates, self.synop_data[self.target_param]], axis=1)

        print("Preparing the dataset")
        for index in tqdm(synop_data_indices):
            self.features.append(self.synop_data.iloc[index:index + self.sequence_length][
                                     list(list(zip(*self.train_params))[1])].to_numpy())
            self.targets.append(labels.iloc[index + self.sequence_length + self.prediction_offset])

        self.features, self.targets = match_gfs_with_synop_sequence(self.features, self.targets,
                                                                    self.target_coords[0],
                                                                    self.target_coords[1],
                                                                    self.prediction_offset,
                                                                    target_param_to_gfs_name_level(self.target_param),
                                                                    return_GFS=False)

        self.data = list(zip(self.features, self.targets))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""

        x, label = self.data[index][0], self.data[index][1]

        return x, label

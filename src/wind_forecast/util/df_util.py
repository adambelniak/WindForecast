from typing import Dict

import pandas as pd
from statsmodels.tsa.seasonal import STL

from wind_forecast.util.common_util import NormalizationType


def resolve_indices(data: pd.DataFrame, indices: [int], length_of_sequence) -> pd.DataFrame:
    # Bear in mind that indices are indices of FIRST data in the sequence. Not all data exist in indices because of that fact.
    all_indices = set(
        [item for sublist in [[index + frame for frame in range(0, length_of_sequence)] for index in indices]
         for item in sublist])
    return data.loc[list(all_indices)]


def normalize_data_for_training(data: pd.DataFrame, data_indices: [int], features: [str],
                                      length_of_sequence: int, normalization_type: NormalizationType
                                      = NormalizationType.STANDARD) -> (pd.DataFrame, Dict, Dict):
    all_relevant_data = resolve_indices(data, data_indices, length_of_sequence)
    final_data = pd.DataFrame()
    mean_or_min_to_return = {}
    std_or_max_to_return = {}
    for feature in features:
        series_to_normalize = all_relevant_data[feature]

        values, mean_or_min, std_or_max = get_normalization_values(series_to_normalize, normalization_type)
        final_data[feature] = values
        mean_or_min_to_return[feature] = mean_or_min
        std_or_max_to_return[feature] = std_or_max

    rest_of_data = all_relevant_data.drop(features, axis=1)
    return pd.concat([final_data, rest_of_data], axis=1,
                     join='inner'), mean_or_min_to_return, std_or_max_to_return


def get_normalization_values(data: pd.DataFrame,
                             normalization_type: NormalizationType = NormalizationType.STANDARD) -> (pd.DataFrame, float, float):
    if normalization_type == NormalizationType.STANDARD:
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        return (data - data_mean) / data_std, data_mean, data_std
    else:
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        return (data - data_min) / (data_max - data_min), data_min, data_max


def decompose_data(data: pd.DataFrame, features: list):
    for feature in features:
        series = data[feature]
        stl = STL(series, seasonal=35, period=24, trend=81, low_pass=25)
        res = stl.fit(inner_iter=1, outer_iter=10)
        O, T, S, R = res.observed, res.trend, res.seasonal, res.resid
        data[f"{feature}_T"] = T
        data[f"{feature}_S"] = S
        data[f"{feature}_R"] = R
        data.drop(columns=[feature], inplace=True)
    return data
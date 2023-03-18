from typing import Dict, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from synop.consts import CLOUD_COVER, LOWER_CLOUDS, CLOUD_COVER_MAX
from wind_forecast.util.common_util import NormalizationType


def resolve_indices(data: pd.DataFrame, indices: [int], length_of_sequence) -> pd.DataFrame:
    # Bear in mind that indices are indices of FIRST data in the sequence. Not all data exist in indices because of that fact.
    all_indices = set(
        [item for sublist in [[index + frame for frame in range(0, length_of_sequence)] for index in indices]
         for item in sublist])
    return data.loc[list(all_indices)]


def normalize_data_for_training(data: pd.DataFrame, data_indices: [int], features: [str],
                                length_of_sequence: int,
                                normalization_type: NormalizationType = NormalizationType.STANDARD) \
        -> (pd.DataFrame, Union[Dict, None], Union[Dict, None], Union[Dict, None], Union[Dict, None]):
    all_relevant_data = resolve_indices(data, data_indices, length_of_sequence)
    final_data = pd.DataFrame()
    mean_to_return = {}
    std_to_return = {}
    min_to_return = {}
    max_to_return = {}
    for feature in features:
        series_to_normalize = all_relevant_data[feature]

        if feature in [LOWER_CLOUDS[1], CLOUD_COVER[1]]:
            min = 0
            max = CLOUD_COVER_MAX
            mean = std = None
            values = np.array(series_to_normalize.values) / max
        else:
            values, mean, std, min, max = get_normalization_values(series_to_normalize, normalization_type)
        final_data[feature] = values
        mean_to_return[feature] = mean
        std_to_return[feature] = std
        min_to_return[feature] = min
        max_to_return[feature] = max

    rest_of_data = all_relevant_data.drop(features, axis=1)
    return pd.concat([final_data, rest_of_data], axis=1,
                     join='inner'), mean_to_return, std_to_return, min_to_return, max_to_return


def normalize_data_for_test(data: pd.DataFrame, features: [str], mean_or_min: dict, std_or_max: dict,
                            normalization_type: NormalizationType = NormalizationType.STANDARD) -> pd.DataFrame:
    final_data = pd.DataFrame()
    for feature in features:
        series_to_normalize = pd.to_numeric(data[feature])
        if feature in [LOWER_CLOUDS[1], CLOUD_COVER[1]]:
            max = CLOUD_COVER_MAX
            values = np.array(series_to_normalize.values) / max
        else:
            if normalization_type == NormalizationType.STANDARD:
                values = (series_to_normalize - mean_or_min[feature]) / std_or_max[feature]
            else:
                values = (series_to_normalize - mean_or_min[feature]) / (std_or_max[feature] - mean_or_min[feature])
        final_data[feature] = values

    rest_of_data = data.drop(features, axis=1)
    return pd.concat([final_data, rest_of_data], axis=1, join='inner')


def get_normalization_values(data: pd.DataFrame,
                             normalization_type: NormalizationType = NormalizationType.STANDARD) \
        -> (pd.DataFrame, Union[float, None], Union[float, None], Union[float, None], Union[float, None]):
    if normalization_type == NormalizationType.STANDARD:
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        return (data - data_mean) / data_std, data_mean, data_std, None, None
    else:
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        return (data - data_min) / (data_max - data_min), None, None, data_min, data_max


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

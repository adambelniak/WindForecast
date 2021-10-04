import math
from enum import Enum

import numpy as np
import pytz
from torch.utils.data import Subset, random_split
from tqdm import tqdm


def prep_zeros_if_needed(value: str, number_of_zeros: int):
    for i in range(number_of_zeros - len(value) + 1):
        value = '0' + value
    return value


def utc_to_local(date):
    return date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Warsaw')).replace(tzinfo=None)


def declination_of_earth(date):
    day_of_year = date.timetuple().tm_yday
    return 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + day_of_year) / 365.0))


def split_dataset(dataset, val_split=0.2, chunk_length=20, sequence_length=None):
    """ This method splits dataset in a random manner and ensures, that for sequential processing
        there will be no frames from a training dataset in a validation dataset by choosing chunk_length consecutive samples
        from validation dataset and removing sequence_length next samples from training dataset.
    """
    length = len(dataset)
    if sequence_length is None:
        # just split randomly
        return random_split(dataset, [length - (int(length * val_split)), int(length * val_split)])

    assert chunk_length < sequence_length

    val_length = math.floor(length * val_split)
    val_ranges = val_length // chunk_length
    rest = val_length - val_ranges * chunk_length
    train_indexes = np.arange(length).tolist()
    val_indexes_to_choose_from = train_indexes
    val_indexes = []

    def do_random_choice(train_indexes, val_indexes, val_indexes_to_choose_from, choice_length):
        index = np.random.choice(val_indexes_to_choose_from)
        chosen_indexes = np.arange(index, index + choice_length).tolist()
        val_indexes.extend(chosen_indexes)
        val_indexes_to_choose_from = [i for i in val_indexes_to_choose_from if i not in chosen_indexes]
        chosen_indexes.extend(np.arange(chosen_indexes[-1] + 1, chosen_indexes[-1] + sequence_length))
        chosen_indexes.extend(np.arange(chosen_indexes[0] - sequence_length, chosen_indexes[0] - 1))
        train_indexes = [i for i in train_indexes if i not in chosen_indexes]
        return train_indexes, val_indexes, val_indexes_to_choose_from

    for _ in tqdm(range(val_ranges)):
        train_indexes, val_indexes, val_indexes_to_choose_from = do_random_choice(train_indexes, val_indexes, val_indexes_to_choose_from, chunk_length)

    if rest > 0:
        train_indexes, val_indexes, val_indexes_to_choose_from = do_random_choice(train_indexes, val_indexes, val_indexes_to_choose_from, rest)

    return Subset(dataset, train_indexes), Subset(dataset, val_indexes)


class NormalizationType(Enum):
    STANDARD = 0
    MINMAX = 1
import math
from enum import Enum
from typing import Sequence

from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_run import Run
from pathlib import Path
import errno

import os
import numpy as np
import pytz
from torch.utils.data import Subset, random_split, Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm

wandb_logger: WandbLogger

def prep_zeros_if_needed(value: str, number_of_zeros: int):
    for i in range(number_of_zeros - len(value) + 1):
        value = '0' + value
    return value


def utc_to_local(date):
    return date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Europe/Warsaw')).replace(tzinfo=None)


def local_to_utc(date):
    tz = pytz.timezone('Europe/Warsaw')
    return tz.normalize(tz.localize(date)).astimezone(pytz.utc)


def declination_of_earth(date):
    day_of_year = date.timetuple().tm_yday
    return 23.45 * np.sin(np.deg2rad(360.0 * (283.0 + day_of_year) / 365.0))


class CustomSubset(Subset):

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        super().__init__(dataset, indices)
        self.mean = dataset.mean
        self.std = dataset.std


def split_dataset(dataset, val_split=0.2, chunk_length=20, sequence_length=None):
    """ Splits dataset in a random manner and ensures, that for sequential processing
        there will be no frames from a training dataset in a validation dataset by choosing chunk_length consecutive samples
        from validation dataset and removing sequence_length following and previous samples from training dataset.
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
    val_indexes_to_choose_from = [i for i in train_indexes if i <= len(train_indexes) - chunk_length]
    val_indexes = []

    def do_random_choice(train_indexes, val_indexes, val_indexes_to_choose_from, choice_length):
        index = np.random.choice(val_indexes_to_choose_from)

        # save indexes that will be included in a validation set
        chosen_indexes = np.arange(index, index + choice_length).tolist()
        val_indexes.extend(chosen_indexes)

        # save indexes which can still be chosen
        val_indexes_to_choose_from = [i for i in val_indexes_to_choose_from if i not in chosen_indexes and
                                      i not in np.arange(chosen_indexes[0] - chunk_length, chosen_indexes[0] - 1)]

        # remove indexes from train dataset
        chosen_indexes.extend(np.arange(chosen_indexes[0] - sequence_length, chosen_indexes[0] - 1))
        chosen_indexes.extend(np.arange(chosen_indexes[-1] + 1, chosen_indexes[-1] + sequence_length))
        train_indexes = [i for i in train_indexes if i not in chosen_indexes]
        return train_indexes, val_indexes, val_indexes_to_choose_from

    print("Splitting dataset into train and val datasets")
    for _ in tqdm(range(val_ranges)):
        train_indexes, val_indexes, val_indexes_to_choose_from = do_random_choice(train_indexes, val_indexes, val_indexes_to_choose_from, chunk_length)

    if rest > 0:
        train_indexes, val_indexes, val_indexes_to_choose_from = do_random_choice(train_indexes, val_indexes, val_indexes_to_choose_from, rest)


    return CustomSubset(dataset, train_indexes), CustomSubset(dataset, val_indexes)


def get_pretrained_artifact_path(pretrained_artifact_path: str, wandb_logger: WandbLogger):
    run: Run = wandb_logger.experiment  # type: ignore

    if pretrained_artifact_path is None:
        raise ValueError('pretrained_artifact_path must be set to use pretrained artifact!')

    wandb_prefix = 'wandb://'

    if pretrained_artifact_path.startswith(wandb_prefix):
        run_name = os.getenv("RUN_NAME")
        # Resuming from wandb artifacts, e.g.:
        # resume_checkpoint: wandb://WANDB_LOGIN/WANDB_PROJECT_NAME/ARTIFACT_NAME:v0@checkpoint.ckpt
        artifact_root = f'{os.getenv("RESULTS_DIR")}/{os.getenv("WANDB_PROJECT")}/{run_name}/artifacts'

        path = pretrained_artifact_path[len(wandb_prefix):]
        artifact_path, checkpoint_path = path.split('@')
        artifact_name = artifact_path.split('/')[-1].replace(':', '-')

        os.makedirs(f'{artifact_root}/{artifact_name}', exist_ok=True)

        artifact = run.use_artifact(artifact_path, type='model')  # type: ignore
        artifact.download()  # type: ignore

        pretrained_artifact_path = f'artifacts/{artifact_name}/{checkpoint_path}'

    if not Path(pretrained_artifact_path).exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pretrained_artifact_path)

    return pretrained_artifact_path


class NormalizationType(Enum):
    STANDARD = 0
    MINMAX = 1

import math
from enum import Enum
from pathlib import Path
from typing import Sequence
import os
import numpy as np
import pytz
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Subset, random_split, Dataset
from torch.utils.data.dataset import T_co
from tqdm import tqdm
from wandb.sdk.wandb_run import Run
import errno

wandb_logger: WandbLogger = WandbLogger(
    project=os.getenv('WANDB_PROJECT'),
    entity=os.getenv('WANDB_ENTITY'),
    name=os.getenv('RUN_NAME'),
    save_dir=os.getenv('RUN_DIR'),
    log_model=os.getenv('LOG_MODEL') == 'True'
)


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


def basic_split_randomly(dataset, val_split: float, test_split: float):
    length = len(dataset)
    train_dataset, test_dataset = random_split(dataset, [length - (int(length * test_split)), int(length * test_split)])
    if val_split == 0:
        return train_dataset, test_dataset
    length = len(train_dataset)
    train_dataset, val_dataset = random_split(dataset, [length - (int(length * val_split)), int(length * val_split)])
    return train_dataset, val_dataset, test_dataset


def sequence_aware_split_randomly(dataset, val_split: float, test_split: float, chunk_length: int,
                                  sequence_length: int):
    if test_split == 0:
        return dataset

    def do_random_choice(train_indexes, test_indexes, test_indexes_to_choose_from, choice_length):
        index = np.random.choice(test_indexes_to_choose_from)

        # save indexes that will be included in a validation set
        chosen_indexes = np.arange(index, index + choice_length).tolist()
        test_indexes.extend(chosen_indexes)

        # save indexes which can still be chosen
        test_indexes_to_choose_from = [i for i in test_indexes_to_choose_from if i not in chosen_indexes and
                                       i not in np.arange(chosen_indexes[0] - chunk_length, chosen_indexes[0] - 1)]

        # remove indexes from train dataset
        chosen_indexes.extend(np.arange(chosen_indexes[0] - sequence_length, chosen_indexes[0] - 1))
        chosen_indexes.extend(np.arange(chosen_indexes[-1] + 1, chosen_indexes[-1] + sequence_length))
        train_indexes = [i for i in train_indexes if i not in chosen_indexes]
        return train_indexes, test_indexes, test_indexes_to_choose_from

    length = len(dataset)
    assert chunk_length < sequence_length, "Chunk length must be smaller than sequence length for chunk random dataset split"

    test_length = math.floor(length * test_split)
    test_ranges = test_length // chunk_length
    rest = test_length - test_ranges * chunk_length
    train_indexes = np.arange(length).tolist()
    test_indexes_to_choose_from = [i for i in train_indexes if i <= len(train_indexes) - chunk_length]
    test_indexes = []

    print("Splitting dataset into train and test/val datasets")
    for _ in tqdm(range(test_ranges)):
        train_indexes, test_indexes, test_indexes_to_choose_from = do_random_choice(train_indexes, test_indexes,
                                                                                    test_indexes_to_choose_from,
                                                                                    chunk_length)

    if rest > 0:
        train_indexes, test_indexes, test_indexes_to_choose_from = do_random_choice(train_indexes, test_indexes,
                                                                                    test_indexes_to_choose_from, rest)

    if val_split == 0:
        return CustomSubset(dataset, train_indexes), CustomSubset(dataset, test_indexes)

    length = len(train_indexes)
    val_length = math.floor(length * val_split)
    val_ranges = val_length // chunk_length
    rest = val_length - val_ranges * chunk_length
    val_indexes_to_choose_from = [i for i in train_indexes if i <= train_indexes[-1] - chunk_length]
    val_indexes = []

    for _ in tqdm(range(val_ranges)):
        train_indexes, test_indexes, val_indexes_to_choose_from = do_random_choice(train_indexes, val_indexes,
                                                                                   val_indexes_to_choose_from,
                                                                                   chunk_length)

    if rest > 0:
        train_indexes, test_indexes, val_indexes_to_choose_from = do_random_choice(train_indexes, val_indexes,
                                                                                   val_indexes_to_choose_from, rest)

    return CustomSubset(dataset, train_indexes), CustomSubset(dataset, val_indexes), CustomSubset(dataset, test_indexes)


def split_dataset(dataset, val_split=0, test_split=0.2, chunk_length=20, split_mode: str = 'random',
                  sequence_length=None):
    if split_mode == 'random':
        """ Splits dataset in a random manner and ensures, that for sequential processing
            there will be no frames from a training dataset in a validation dataset by choosing chunk_length consecutive samples
            from validation dataset and removing sequence_length following and previous samples from training dataset.
        """
        if sequence_length is None:
            # just split randomly
            return basic_split_randomly(dataset, val_split, test_split)
        return sequence_aware_split_randomly(dataset, val_split, test_split, chunk_length, sequence_length)

    else:
        if sequence_length is None:
            length = len(dataset)
            indexes = np.arange(length)
            test_indexes = indexes[length - int(test_split * length):]
            non_test_length = length - len(test_indexes)
            val_indexes = indexes[non_test_length - int(val_split * non_test_length):]
            train_indexes = indexes[:non_test_length]
            return CustomSubset(dataset, train_indexes), CustomSubset(dataset, val_indexes), CustomSubset(dataset,
                                                                                                          test_indexes)
        else:
            length = len(dataset)
            indexes = np.arange(length)
            test_indexes = indexes[int(length - test_split * length + sequence_length):]
            non_test_indexes = indexes[:int(length - test_split * length + sequence_length)]
            non_test_length = len(non_test_indexes)
            val_indexes = non_test_indexes[int(non_test_length - val_split * non_test_length + sequence_length):]
            train_indexes = non_test_indexes[:int(non_test_length - val_split * non_test_length)]
            return CustomSubset(dataset, train_indexes), CustomSubset(dataset, val_indexes), \
                   CustomSubset(dataset, test_indexes)


def get_pretrained_artifact_path(pretrained_artifact: str):
    run: Run = wandb_logger.experiment  # type: ignore

    if pretrained_artifact is None:
        raise ValueError('pretrained_artifact must be set to use pretrained artifact!')

    wandb_prefix = 'wandb://'

    if pretrained_artifact.startswith(wandb_prefix):
        run_name = os.getenv("RUN_NAME")
        # Resuming from wandb artifacts, e.g.:
        # resume_checkpoint: wandb://WANDB_LOGIN/WANDB_PROJECT_NAME/ARTIFACT_NAME:v0@checkpoint.ckpt
        artifact_root = f'{os.getenv("RESULTS_DIR")}/{os.getenv("WANDB_PROJECT")}/{run_name}/artifacts'

        path = pretrained_artifact[len(wandb_prefix):]
        artifact_path, checkpoint_path = path.split('@')
        print(artifact_path)
        artifact_name = artifact_path.split('/')[-1].replace(':', '-')
        print(artifact_name)

        os.makedirs(f'artifacts/{artifact_name}', exist_ok=True)

        artifact = run.use_artifact(artifact_path, type='model')  # type: ignore
        artifact.download(root=f'artifacts/{artifact_name}')  # type: ignore

        pretrained_artifact_path = f'artifacts/{artifact_name}/{checkpoint_path}'

    if not Path(pretrained_artifact_path).exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pretrained_artifact_path)

    return pretrained_artifact_path


class NormalizationType(Enum):
    STANDARD = 0
    MINMAX = 1

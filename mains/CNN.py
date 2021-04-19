import argparse
import sys
import os

sys.path.insert(1, '..')

from util.utils import get_available_numpy_files
from generators.MultiChannelDataGeneratorWithEarthDeclination import MultiChannelDataGeneratorWithEarthDeclination
from models.common import plot_history
from models.CNNModel import create_model
from util.config import process_config


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_training_and_validation_generators(IDs, config):
    features, target_attribute, synop_file = config.train_parameters, config.target_parameter, config.synop_file
    length = len(IDs)
    print(length)
    print(IDs[0:6])
    training_data, validation_data = IDs[:int(length * 0.8)], IDs[int(length * 0.8):]
    training_datagen = MultiChannelDataGeneratorWithEarthDeclination(training_data, features, target_attribute, synop_file, dim=(config.data_dim_y, config.data_dim_x), normalize=True)
    validation_datagen = MultiChannelDataGeneratorWithEarthDeclination(validation_data, features, target_attribute, synop_file, dim=(config.data_dim_y, config.data_dim_x), normalize=True)

    return training_datagen, validation_datagen


def train_model(config):
    features, target_attribute, offset, gfs_dir, synop_file = config.train_parameters, config.target_parameter,\
                                                              config.prediction_offset, config.gfs_dataset_dir,\
                                                              config.synop_file

    IDs = get_available_numpy_files(features, offset, gfs_dir)
    training_datagen, validation_datagen = get_training_and_validation_generators(IDs, config)

    model = create_model((config.data_dim_y, config.data_dim_x, len(features)))
    history = model.fit_generator(generator=training_datagen, validation_data=validation_datagen, epochs=40)
    plot_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help='Configuration file path', type=str, default="../config/CNNConfig.json")
    args = parser.parse_args()

    config = process_config(args.config)
    train_model(config)


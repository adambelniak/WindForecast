import argparse
import sys

from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os

sys.path.insert(1, '../..')

from preprocess.gfs.spatial.convolution_preprocess import prepare_training_data
from models.common import GFS_PARAMETERS, plot_history
from preprocess.synop.consts import TEMPERATURE, VELOCITY_COLUMN
from models.ResNet.ResNet_model import create_model
from keras.applications.resnet50 import preprocess_input

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def train_model(training_data):
    x_train, x_valid, y_train, y_valid = train_test_split(training_data[0], training_data[1], test_size=0.2, shuffle=True)
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, preprocessing_function=preprocess_input)
    datagen.fit(x_train)
    # print(datagen.mean)
    train_iterator = datagen.flow(x_train, y_train, batch_size=16)
    val_iterator = datagen.flow(x_valid, y_valid, batch_size=16)
    #
    model = create_model(training_data[0][0].shape)
    history = model.fit_generator(train_iterator, epochs=20, validation_data=val_iterator)
    #
    # # temp_slice = create_single_slice_for_param_and_region(datetime(2015, 2, 9), '00', 3, 'TMP', 'HTGL_2', 56, 48, 13, 26)
    # # print(temp_slice)
    # # clouds_slice = create_single_slice_for_param_and_region(datetime(2015, 2, 9), '0', 3, 'T CDC', 'LCY_0', 56, 48, 13, 26)
    # # check_slice = np.array([np.einsum('kli->lik', np.array([temp_slice]))])
    # # prediction = model.predict_generator(datagen.flow(check_slice))
    # # print(prediction * datagen.std + datagen.mean)
    plot_history(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--years', help='Years of GFS forecasts to take into account.', nargs="+", required=True, type=int)
    parser.add_argument('--gfs_dataset_dir', help='Directory with dataset of gfs forecasts.', type=str, default="D:\\WindForecast\\output_np")
    parser.add_argument('--synop_dataset_path', help='Directory with dataset of gfs forecasts.', type=str, default="../../datasets/synop/KOZIENICE_488_data.csv")

    args = parser.parse_args()

    PARAMETERS = [
        # ("TMP", "HTGL_2"),
        # ("T CDC", "LCY_0")
        ("V GRD", "HTGL_10"),
        ("U GRD", "HTGL_10"),
        ("GUST", "SFC_0")
    ]
    training, labels = prepare_training_data(PARAMETERS, VELOCITY_COLUMN[1], 3, args.years, args.gfs_dataset_dir, args.synop_dataset_path)

    train_model((training, labels))
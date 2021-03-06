from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D


def create_model(input_shape):
    model = Sequential([
        Conv2D(input_shape=input_shape, data_format='channels_first', filters=32, kernel_size=3, padding="same",
               activation="relu"),
        MaxPooling2D(padding="same", data_format='channels_first'),
        Conv2D(input_shape=input_shape, data_format='channels_first', filters=64, kernel_size=3, padding="same",
               activation="relu"),
        MaxPooling2D(padding="same", data_format='channels_first'),
        Conv2D(input_shape=input_shape, data_format='channels_first', filters=128, kernel_size=3, padding="same",
               activation="relu"),
        MaxPooling2D(padding="same",data_format='channels_first'),
        Dense(128)
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    return model

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import Dropout


def create_model(input_shape):
    model = Sequential([
        Dropout(.4, input_shape=input_shape),
        Conv2D(filters=32, kernel_size=3, padding="same",
               activation="swish"),
        MaxPooling2D(padding="same"),
        Dropout(.4, input_shape=input_shape),
        Conv2D(input_shape=input_shape, filters=64, kernel_size=3, padding="same",
               activation="swish"),
        MaxPooling2D(padding="same"),
        Conv2D(input_shape=input_shape, filters=128, kernel_size=3, padding="same",
               activation="swish"),
        MaxPooling2D(padding="same"),
        Flatten(),
        Dense(128, activation="linear"),
        Dense(1, activation="linear")
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    return model

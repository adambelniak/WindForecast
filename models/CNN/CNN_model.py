from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten


def create_model(input_shape):
    model = Sequential([
        Conv2D(input_shape=input_shape, filters=32, kernel_size=3, padding="same",
               activation="relu"),
        MaxPooling2D(padding="same"),
        Conv2D(input_shape=input_shape, filters=64, kernel_size=3, padding="same",
               activation="relu"),
        MaxPooling2D(padding="same"),
        Conv2D(input_shape=input_shape, filters=128, kernel_size=3, padding="same",
               activation="relu"),
        MaxPooling2D(padding="same"),
        Flatten(),
        Dense(128, activation="linear"),
        Dense(1, activation="linear")
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    return model

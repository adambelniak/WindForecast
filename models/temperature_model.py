from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model(input_dim: int):
    model = Sequential([
        Dense(units=32, input_shape=input_dim, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=1)
    ])
    model.summary()

    return model

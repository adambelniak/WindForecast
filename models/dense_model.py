from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model():
    model = Sequential([
        Dense(units=32, input_shape=(7, ), activation='relu'),
        Dense(units=64, activation='relu'),
        # Dense(units=128, activation='relu'),
        # Dense(units=64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

    return model

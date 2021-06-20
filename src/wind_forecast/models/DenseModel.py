from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input, Model


def create_model(input_shape):
    input = Input(input_shape)
    X = Dense(units=32, input_shape=input_shape, activation='relu')(input)
    X = Dense(units=64, activation='relu')(X)
    X = Dense(units=128, activation='relu')(X)
    X = Dense(units=64, activation='relu')(X)
    X = Dense(units=32, activation='relu')(X)
    X = Dense(units=1)(X)
    model = Model(inputs=input, outputs=X, name='Dense')
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

    return model

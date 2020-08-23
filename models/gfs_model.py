from tensorflow import keras
import pandas as pd


def create_model(inputs, gfs_input, learning_rate: float, ):
    auxiliary_input = keras.layers.Input(shape=(gfs_input.shape[1],), name='single_gfs_input')

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    keras.layers.Concatenate()([lstm_out, auxiliary_input])

    outputs = keras.layers.Dense(1)(lstm_out)

    model = keras.Model(inputs=[inputs, auxiliary_input], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    return model


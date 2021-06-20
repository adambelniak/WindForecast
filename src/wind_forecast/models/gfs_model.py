from tensorflow import keras


def create_model(inputs, gfs_input, learning_rate: float, ):
    auxiliary_input = keras.layers.Input(shape=(gfs_input.shape[1],), name='single_gfs_input')

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    conc = keras.layers.Concatenate()([lstm_out, auxiliary_input])
    outputs = keras.layers.Dense(32, activation='relu')(conc)

    outputs = keras.layers.Dense(1)(outputs)

    model = keras.Model(inputs=[inputs, auxiliary_input], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    return model


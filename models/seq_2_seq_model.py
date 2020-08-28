from tensorflow import keras


def create_model(inputs, gfs_input, learning_rate: float, sequence_length: int):
    gfs_input = keras.layers.Input(shape=(gfs_input.shape[1], gfs_input.shape[2]), name='sequence_gfs_input')

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32, )(inputs)
    lstm_lstm_out = keras.layers.LSTM(32)(gfs_input)

    conc = keras.layers.Concatenate()([lstm_out, lstm_lstm_out])
    repeated_output = keras.layers.RepeatVector(sequence_length)(conc)
    decoder_lstm, _, _ = keras.layers.LSTM(32, return_sequences=True, return_state=True)(repeated_output)

    outputs = keras.layers.TimeDistributed(keras.layers.Dense(1))(decoder_lstm)

    model = keras.Model(inputs=[inputs, gfs_input], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    return model


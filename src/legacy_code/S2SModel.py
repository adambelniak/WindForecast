from tensorflow import keras

# Deprecated


def create_model(inputs, gfs_input, learning_rate: float, sequence_length: int):
    gfs_input = keras.layers.Input(shape=(gfs_input.shape[1], gfs_input.shape[2]), name='sequence_gfs_input')

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    encoder_outputs, state_h, state_c = keras.layers.LSTM(32, return_state=True)(inputs)
    gfs_encoder_outputs, gfs_state_h, gfs_state_c= keras.layers.LSTM(32, return_state=True)(gfs_input)

    conc = keras.layers.Concatenate()([gfs_state_h, state_h])
    conc_2 = keras.layers.Concatenate()([state_c, gfs_state_c])
    decoder_inputs = keras.layers.Input(shape=(None, 1))

    decoder_lstm, _, _ = keras.layers.LSTM(64, return_sequences=True, return_state=True)(decoder_inputs, initial_state=[conc, conc_2])

    outputs = keras.layers.Dense(1)(decoder_lstm)

    model = keras.Model(inputs=[inputs, gfs_input, decoder_inputs], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    return model


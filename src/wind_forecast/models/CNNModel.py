from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dropout, concatenate


def create_model(input_shape):
    input = Input(input_shape)
    X = Dropout(.4, input_shape=input_shape)(input)
    X = Conv2D(filters=32, kernel_size=3, padding="same",
               activation="swish")(X)
    X = MaxPooling2D(padding="same")(X)
    X = Dropout(.4, input_shape=input_shape)(X)
    X = Conv2D(input_shape=input_shape, filters=64, kernel_size=3, padding="same",
               activation="swish")(X)
    X = MaxPooling2D(padding="same")(X)
    X = Conv2D(input_shape=input_shape, filters=128, kernel_size=3, padding="same",
               activation="swish")(X)
    X = MaxPooling2D(padding="same")(X)
    X = Flatten()(X)

    day_length_input = Input(1)

    concat_input = concatenate([day_length_input, X])

    X = Dense(129, activation="linear")(concat_input)
    X = Dense(1, activation="linear")(X)
    model = Model(inputs=[input, day_length_input], outputs=X, name='CNN')
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    return model

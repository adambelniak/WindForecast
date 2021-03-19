from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dropout, Dense, Flatten


def create_model(input_shape):
    base_model = ResNet50(include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='linear')(x)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # model.summary()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model

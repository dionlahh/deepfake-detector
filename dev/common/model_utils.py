from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import efficientnet as efn

# Import keras applications for model input preprocessing functions
import tensorflow.keras.applications as app
import gc
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer
from keras import optimizers


def model_selection(model_type: str):
    models = {
        "EfficientNetB0": efn.EfficientNetB0,
        "EfficientNetB1": efn.EfficientNetB1,
        "EfficientNetB2": efn.EfficientNetB2,
        "EfficientNetB3": efn.EfficientNetB3,
    }
    return models.get(model_type, None)

def input_size(model_type: str):
    models = {
        "EfficientNetB0": 224,
        "EfficientNetB1": 240,
        "EfficientNetB2": 260,
        "EfficientNetB3": 300,
    }
    return models.get(model_type, None)

# Model Building
def build_model(model_type: str, input_shape):
    model = Sequential()

    # Add an InputLayer with the specified input shape
    model.add(InputLayer(input_shape=input_shape))

    # Use model_selection to get the EfficientNet model based on model_type
    efficientnet_model = model_selection(model_type)(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,  # Specify the input shape for EfficientNet
        pooling="avg",
    )

    model.add(efficientnet_model)

    model.add(Flatten())
    model.add(Dense(2, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.layers[1].trainable = True  # train all the layers except the InputLayer
    model.summary()

    optimizer = optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()

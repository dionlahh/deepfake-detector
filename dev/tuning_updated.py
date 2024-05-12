import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, InputLayer, Dropout
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
import keras_tuner as kt
from common.model_utils import model_selection, input_size
from tensorflow.keras.applications import efficientnet as efn
import gc
import os
import time
import threading
from tensorflow.python.client import device_lib
import tensorflow as tf


#TODO: Change accordingly
LANDMARK = True 
RESUMING_JOB = True
BUILD_ONLY = False
MODEL_TYPE = "EfficientNetB1" 
PROJECT_NAME = f"{MODEL_TYPE} Tuned"

# Constant fields
BATCH_SIZE = 16

# Folder Paths
landmark_models_folder = "/home/FYP/dion0020/deepfake-detector/dev/Landmark Models"
models_folder = "/home/FYP/dion0020/deepfake-detector/dev/Benchmark Models"
data_folder = "/home/FYP/dion0020/deepfake-detector/dev/Training Data"
landmark_data_folder = "/home/FYP/dion0020/deepfake-detector/dev/Training Landmarks Data"
tuning_folder = "/home/FYP/dion0020/deepfake-detector/dev/Tuning"
landmark_tuning_folder = "/home/FYP/dion0020/deepfake-detector/dev/Landmark Tuning"

size = input_size(MODEL_TYPE)
imgsize = (size, size)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# Function to save the tuner state
def save_tuner_state():
    tuner.save('tuner_state')

# Model builder function for Keras Tuner
def model_builder(hp):
    # Create a Sequential model
    
    built_model = Sequential()
    
    built_model.add(InputLayer(input_shape=(size, size, 3)))

    # Add a base model
    built_model.add(
        model_selection(MODEL_TYPE)(
        include_top=False,
        weights="imagenet",
        input_shape=(size, size, 3),  # Specify the input shape for EfficientNet
        pooling="avg",
    )
    )

    built_model.add(Flatten())

    # Tune the number of neurons in the first Dense layer
    hp_units = hp.Choice('units', values=[2, 32, 64, 128]) # start at 32
    hp_activation = hp.Choice('activation_function', values=['relu', 'tanh'])
    built_model.add(Dense(units=hp_units, activation=hp_activation))

    # Tune the learning rate for Adam optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[0.001, 0.0001, 0.00001])
    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Tune the dropout layer
    hp_dropout = hp.Choice("dropout", values=[0.1, 0.2, 0.3])
    built_model.add(Dropout(hp_dropout))

    # Output layer with 2 units for binary classification
    built_model.add(Dense(2, activation="softmax"))

    # built_model.layers[0].trainable = False
    built_model.layers[1].trainable = True  # train all the layers except the InputLayer


    # Compile the model
    built_model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return built_model

if __name__ == "__main__":
    print(get_available_gpus())
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Initialize the ImageDataGenerator object
    if LANDMARK == False:
        trainpath = os.path.join(".", data_folder, "Training")
        valpath = os.path.join(".", data_folder, "Validation")
        tuning_dir = tuning_folder
        model_dir = models_folder
    else:
        trainpath = os.path.join(".", landmark_data_folder, "Training")
        valpath = os.path.join(".", landmark_data_folder, "Validation")
        tuning_dir = landmark_tuning_folder
        model_dir = landmark_models_folder

    train_datagen = ImageDataGenerator(
        preprocessing_function=efn.preprocess_input
    )
    train_gen = train_datagen.flow_from_directory(
        trainpath, target_size=imgsize, batch_size=BATCH_SIZE, class_mode="categorical"
    )
    val_gen = train_datagen.flow_from_directory(
        valpath, target_size=imgsize, batch_size=BATCH_SIZE, class_mode="categorical"
    )

    tuner = kt.tuners.Hyperband(
        model_builder,
        objective="val_categorical_accuracy",
        max_epochs=20,  # Maximum number of epochs for each configuration
        factor=3,  # Reduction factor for the number of configurations and epochs
        directory=tuning_dir,
        project_name=PROJECT_NAME,
        overwrite= not RESUMING_JOB,  # Set to False when resuming
    )

    if os.path.exists('tuner_state'):
        tuner.reload('tuner_state')
    else:
        gc.collect()
        k.clear_session()

    # Search for the best hyperparameters
    if not BUILD_ONLY:
        tuner.search(
            train_gen,
            validation_data=val_gen,
        )

    # tuner.save('tuner_state')

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best params:", best_hps)

    best_hyperparameters_file = f"{tuning_dir}/{PROJECT_NAME}/{MODEL_TYPE}_best_hyperparameters.txt"
    with open(best_hyperparameters_file, "w") as f:
        for hp_name, hp_value in best_hps.values.items():
            f.write(f"{hp_name}: {hp_value}\n")

    print("Starting model training with best params")

    cb_early_stopper = EarlyStopping(monitor="val_loss", patience=5)
    cb_checkpointer = ModelCheckpoint(
        filepath=os.path.join(".", model_dir, f"{PROJECT_NAME}.hdf5"),
        monitor="val_loss",
        save_best_only=True,
        mode="auto",
    )

    # Build and train the model with the best hyperparameters
    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[cb_checkpointer, cb_early_stopper],
    )

    print(best_model.summary())

    print("Saving model...")
    best_model.save(f"{model_dir}/{PROJECT_NAME}.hdf5")
    print("Saved!")
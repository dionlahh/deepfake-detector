import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import efficientnet as efn
from common.model_utils import input_size
from sklearn import metrics
import os
import json

# Function to load TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to preprocess image
def preprocess_image(image_path, img_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# PLEASE PUT IN CORRECT FOLDER DIRECTORIES
data_folder = ".../dev/Training Data"
landmark_data_folder = ".../dev/Training Landmarks Data"
result_folder = ".../dev/Result"
landmark_result_folder = ".../dev/Landmark Result"
models_folder = ".../dev/Benchmark Models"
landmark_models_folder = ".../dev/Landmark Models"
tuning_folder = ".../dev/Tuning"
landmark_tuning_folder = ".../dev/Landmark Tuning"

model_path = "tflite/BaseB0.tflite"
model_type = "BaseB0"
LANDMARK = False

if __name__ == "__main__":
    if LANDMARK:
        tuning_dir = landmark_tuning_folder
        data_dir = landmark_data_folder
        model_dir = landmark_models_folder
        result_dir = landmark_result_folder
    else:
        tuning_dir = tuning_folder
        data_dir = data_folder
        model_dir = models_folder
        result_dir = result_folder

    test_path = os.path.join(".", data_dir, "Test")
    model_dir = ".../dev/tflite"
    model_path = os.path.join(model_dir, "BaseB0.tflite")
    model = load_tflite_model(model_path)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model
    test_img_gen = ImageDataGenerator(preprocessing_function=None)  # No preprocessing for TFLite models

    img_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])  # Define your function to get the input size based on the model type

    test_generator = test_img_gen.flow_from_directory(
        directory=test_path,
        target_size=img_size,
        batch_size=1,  # Set batch size to 1 for TFLite inference
        class_mode=None,
        shuffle=False,
        seed=123,
    )

    # Perform inference and evaluate model
    predicted_class_indices = []
    for i in range(len(test_generator)):
        image_batch = test_generator.next()
        model.set_tensor(model.get_input_details()[0]['index'], image_batch)
        model.invoke()
        output_data = model.get_tensor(model.get_output_details()[0]['index'])
        predicted_class_indices.extend(np.argmax(output_data, axis=1))

    # Get ground truth labels
    y = test_generator.classes

    # Calculate evaluation metrics
    accuracy = metrics.accuracy_score(y, predicted_class_indices)
    precision = metrics.precision_score(y, predicted_class_indices, average="binary")
    recall = metrics.recall_score(y, predicted_class_indices, average="binary")
    auc = metrics.roc_auc_score(y, predicted_class_indices)

    print("Model Accuracy:", accuracy)
    print("Model Precision:", precision)
    print("Model Recall:", recall)
    print("AUC:", auc)

    # Save results to CSV
    result_dir = ".../dev/tflite/results"
    filenames = [os.path.split(i)[1] for i in test_generator.filenames]
    actual_labels = [os.path.split(i)[0] for i in test_generator.filenames]
    actual_labels = ["0" if label == "manipulated" else "1" for label in actual_labels]

    results_df = pd.DataFrame({
        "id": filenames,
        "actual label": actual_labels,
        "pred label": predicted_class_indices
    })

    os.makedirs(result_dir, exist_ok=True)
    results_df.to_csv(os.path.join(result_dir, model_type + "_out.csv"))

    # Save model results to JSON
    models_result = {
        model_type: {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "AUC": auc
        }
    }

    with open(os.path.join(result_dir, "model_results.json"), "w") as outfile:
        json.dump(models_result, outfile)

    # Save model prediction results to CSV
    model_prediction_results = pd.DataFrame.from_dict(models_result, orient="index")
    model_prediction_results.to_csv(os.path.join(result_dir, "prediction_results.csv"))
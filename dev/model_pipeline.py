from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
import facial_detector
from common.model_utils import input_size
import numpy as np
import os
import json

base_model_paths = {
    "EfficientNetB0": "/Users/dion/Github/Y4_Projects/FYP/dev/Benchmark Models/EfficientNetB0.hdf5",
    "EfficientNetB1": "/Users/dion/Github/Y4_Projects/FYP/dev/Benchmark Models/EfficientNetB1.hdf5",
    "EfficientNetB2": "/Users/dion/Github/Y4_Projects/FYP/dev/Benchmark Models/EfficientNetB2.hdf5",
    "EfficientNetB3": "/Users/dion/Github/Y4_Projects/FYP/dev/Benchmark Models/EfficientNetB3.hdf5",
}

landmark_model_paths = {
    "EfficientNetB0": "/Users/dion/Github/Y4_Projects/FYP/dev/Landmark Models/EfficientNetB0.hdf5",
    "EfficientNetB1": "/Users/dion/Github/Y4_Projects/FYP/dev/Landmark Models/EfficientNetB1.hdf5",
    "EfficientNetB2": "/Users/dion/Github/Y4_Projects/FYP/dev/Landmark Models/EfficientNetB2.hdf5",
    "EfficientNetB3": "/Users/dion/Github/Y4_Projects/FYP/dev/Landmark Models/EfficientNetB3.hdf5",
}


class pipeline:

    def __init__(self, landmark, model_path, model_type):
        self.isLandmark = landmark
        self.model = load_model(model_path)
        self.facial_detector = facial_detector.FaceDetector()
        self.model_type = model_type

    def get_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        return frame

    def run_face_detector(self, frame):
        faces = self.facial_detector.run_face_detector(frame)
        return faces

    def run_face_landmark_detector(self, frame):
        faces = self.facial_detector.run_face_detector(frame)
        landmarked_faces = []
        for i, face in enumerate(faces):
            landmarked_faces.append(self.facial_detector.plot_facial_landmarks(face))
        return faces, landmarked_faces

    def preprocess_and_resize(self, face, target_size):
        preprocessed_face = preprocess_input(face)
        preprocessed_face = cv2.resize(preprocessed_face, (target_size, target_size))
        preprocessed_face = preprocessed_face.reshape(1, target_size, target_size, 3)
        return preprocessed_face

    def run_model(self, face):
        imgsize = input_size(self.model_type)
        processed_face = self.preprocess_and_resize(face, imgsize)

        return self.model.predict(processed_face)

    def format_predictions(self, predictions):
        class_labels = ["Manipulated", "Original"]
        formatted_predictions = []

        for pred_array in predictions:
            label_index = np.argmax(pred_array)
            label = class_labels[label_index]
            confidence = pred_array[0][label_index]
            # print("DEBUG: ", pred_array)
            formatted_predictions.append((label, confidence))

        return formatted_predictions

    def format_predictions_for_submission(self, predictions):
        class_labels = ["fake", "real"]
        formatted_predictions = []
        label = None

        for pred_array in predictions:
            label_index = np.argmax(pred_array)
            label = class_labels[label_index]
            confidence = pred_array[0][label_index]
            # print("DEBUG: ", pred_array)
            formatted_predictions.append((label, confidence))

        return "real" if label == None else label

    def print_prediction(self, predictions):
        for label, confidence in predictions:
            print(f"Label: {label}\nConfidence: {confidence}")

    def run_pipeline(self, frame_path):
        frame = self.get_frame(frame_path)

        if self.isLandmark:
            original_faces, faces = self.run_face_landmark_detector(frame)
            results = [-1 for i in faces]

        else:
            faces = self.run_face_detector(frame)
            results = [-1 for i in faces]

        for i, face in enumerate(faces):
            label = self.run_model(face)
            results[i] = label

        # self.print_prediction(self.format_predictions(results))
        print(f"debugg: {self.format_predictions(results)[0][0]}")

        if self.isLandmark:
            return original_faces, self.format_predictions(results)
        else:
            return faces, self.format_predictions(results)

    def run_pipeline_on_dir(self, image_dir):

        if not image_dir.endswith("/"):
            image_dir += "/"

        files = os.listdir(image_dir)
        png_files = [file for file in files if file.endswith(".png")]
        results_dict = {}

        for image in png_files:

            frame = self.get_frame(image_dir + image)

            if self.isLandmark:
                original_faces, faces = self.run_face_landmark_detector(frame)
                results = [-1 for i in faces]

            else:
                faces = self.run_face_detector(frame)
                results = [-1 for i in faces]

            for i, face in enumerate(faces):
                label = self.run_model(face)
                results[i] = label

            # self.print_prediction(self.format_predictions(results))
            pred_label = self.format_predictions_for_submission(results)
            results_dict[image] = pred_label

        return results_dict

        # if self.isLandmark:
        #     return original_faces, self.format_predictions(results)
        # else:
        #     return faces, self.format_predictions(results)


def main(isLandmark, model_choice, model_path, image):

    # if isLandmark:
    #     model_path = landmark_model_paths[model_choice]
    # else:
    #     model_path = base_model_paths[model_choice]

    base_pipline = pipeline(
        isLandmark,
        model_path,
        model_choice,
    )

    return base_pipline.run_pipeline(image)


def submissionTest(isLandmark, model_choice, model_path, image_dir):

    base_pipline = pipeline(
        isLandmark,
        model_path,
        model_choice,
    )

    return base_pipline.run_pipeline_on_dir(image_dir)


if __name__ == "__main__":

    model_dir = "/Users/dion/Github/Y4_Projects/FYP/dev/Tuned Models"
    model_types = ["Base Models", "Landmark Models"]
    model_names = ["EfficientNetB0 Tuned", "EfficientNetB1 Tuned"]

    result_dir = (
        "/Users/dion/Github/Y4_Projects/FYP/dev/faceforensics_benchmark_results"
    )

    # test_image_dir = "/Users/dion/Github/Y4_Projects/FYP/dev/untitled folder"
    test_image_dir = (
        "/Users/dion/Github/Y4_Projects/FYP/dev/faceforensics_benchmark_images"
    )

    for model_type in model_types:
        for model_name in model_names:
            print(f"===== Currently testing {model_type}/{model_name} =====")
            files = os.listdir(test_image_dir)
            result_dict = {}
            file_name = (
                f"{result_dir}/{model_type.split()[0]}_{model_name.split()[0]}.json"
            )
            for image in files:
                image_path = f"{test_image_dir}/{image}"
                _, result = main(
                    isLandmark=False,
                    model_choice=model_name.split()[0],
                    model_path=f"{model_dir}/{model_type}/{model_name}.hdf5",
                    image=image_path,
                )
                result_dict[image] = result[0][0]

            with open(file_name, "w") as file:
                # Write the dictionary to the file in JSON format
                json.dump(result_dict, file)

                # pred_label = result[0][0]
                # print(f"{image}:{result[0][0]}")
            # result = submissionTest(
            #     isLandmark=False,
            #     model_choice=model_name.split()[0],
            #     model_path=f"{model_dir}/{model_type}/{model_name}.hdf5",
            #     image_dir=test_image_dir,
            # )
            # file_name = (
            #     f"{result_dir}/{model_type.split()[0]}_{model_name.split()[0]}.json"
            # )

            # with open(file_name, "w") as file:
            #     # Write the dictionary to the file in JSON format
            #     json.dump(result, file)

    # print(result)

    # base_pipline = pipeline(
    #     False,
    #     "/Users/dion/Github/Y4_Projects/FYP/dev/Benchmark Models/EfficientNetB0.hdf5",
    #     "EfficientNetB0",
    # )

    # # faces, results = base_pipline.run_pipeline(
    # #     "/Users/dion/Github/Y4_Projects/FYP/dev/Split Data/Test/manipulated/frames/faked_frame_0_35.jpg"
    # # )

    # faces, results = base_pipline.run_pipeline(
    #     "/Users/dion/Github/Y4_Projects/FYP/dev/Split Data/Test/original/frames/original_frame_0_35.jpg"
    # )

import cv2
import dlib
import numpy as np
import os

def predict_facial_landmarks_dlib(image_path, output_folder):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained facial landmarks predictor from DLIB
    predictor_path = "shape_predictor_68_face_landmarks.dat"  # You need to download this file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Detect faces in the image
    faces = detector(gray)

    if not faces:
        print("No faces detected in", image_path)
        return None

    # Get facial landmarks for the first face
    shape = predictor(gray, faces[0])

    # Convert the landmarks to a NumPy array
    landmarks = np.array([[point.x, point.y] for point in shape.parts()])

    # Draw circles on the image at the landmark points
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # You can adjust the color and radius as needed

    # Save the image with landmarks to the output folder
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

    return landmarks

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            predict_facial_landmarks_dlib(image_path, output_folder)
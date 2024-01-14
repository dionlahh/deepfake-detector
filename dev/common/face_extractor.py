import cv2
import os
import dlib
import numpy as np
from PIL import Image


def has_face_haar_cascade(frame):
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Return True if at least one face is detected
    return len(faces) > 0


def verify_face_dlib(folder_path, show_progress=True):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Get the total number of images in the folder
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith((".jpg", ".jpeg", ".png", ".gif"))
    ]
    total_images = len(image_files)

    # Counter for the processed images
    processed_images = 0

    # Iterate through all files in the specified folder
    for filename in image_files:
        file_path = os.path.join(folder_path, filename)

        # Load the image using PIL (Python Imaging Library)
        img = Image.open(file_path)

        # Convert the PIL image to a numpy array for dlib
        img_array = np.array(img)

        # Use the face detector to find faces in the image
        faces = face_detector(img_array)

        # If no faces are detected, delete the image
        if len(faces) == 0:
            if show_progress == True:
                print(f"No face detected in {filename}. Deleting the image.")
            os.remove(file_path)

        # Update progress
        processed_images += 1
        progress_percentage = (processed_images / total_images) * 100
        if show_progress == True:
            print(
                f"Progress: {processed_images}/{total_images} images processed ({progress_percentage:.2f}%)"
            )


def save_faces_from_directory(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Detect faces in each frame in the input folder
    for file_name in os.listdir(input_folder):
        frame_path = os.path.join(input_folder, file_name)

        # Read the frame
        frame = cv2.imread(frame_path)

        # Check if the frame has a face using the has_face function
        if has_face_haar_cascade(frame):
            # Get the base name of the original frame file
            image_name = os.path.splitext(file_name)[0]

            # Detect faces in the frame
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            # Crop and save the image for each detected face
            for i, (x, y, w, h) in enumerate(faces):
                face_image = frame[y : y + h, x : x + w]
                cropped_frame_filename = f"{output_folder}/{image_name}_face_{i}.jpg"
                cv2.imwrite(cropped_frame_filename, face_image)


def extract_face(input_dir, output_dir):
    save_faces_from_directory(input_dir, output_dir)
    verify_face_dlib(output_dir)


def save_faces_from_image(image_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image has a face using the has_face function
    if has_face_haar_cascade(image):
        # Get the base name of the original image file
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Detect faces in the image
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        # Crop and save the image for each detected face
        for i, (x, y, w, h) in enumerate(faces):
            face_image = image[y : y + h, x : x + w]
            cropped_image_filename = f"{output_folder}/{image_name}_face_{i}.jpg"
            cv2.imwrite(cropped_image_filename, face_image)

        # Verify faces using dlib
        verify_face_dlib(output_folder, show_progress=False)


def count_jpg_files(folder_path):
    """
    Count the number of JPG files in the specified folder.

    Parameters:
    - folder_path (str): The path to the folder.

    Returns:
    - int: The number of JPG files in the folder.
    """
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return 0

    # List all files in the folder
    all_files = os.listdir(folder_path)

    # Count the number of JPG files
    jpg_files = [file for file in all_files if file.lower().endswith(".jpg")]

    return len(jpg_files)

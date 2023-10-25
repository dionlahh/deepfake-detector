import cv2
import dlib
import os


def has_face(frame):
    # Load the Dlib face detector
    detector = dlib.get_frontal_face_detector()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray_frame)

    # Return True if at least one face is detected
    return len(faces) > 0


def extract_frames(input_folder, output_folder, faked=True):
    manipulated = "faked" if faked else "original"
    video_index = 0

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files and subdirectories in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if it's a video file
            if file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                print(f"Processing video: {file_name} | ", end="")

                # Open the video file
                cap = cv2.VideoCapture(file_path)

                # Get the total number of frames in the video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Calculate frame indices for 25%, 50%, and 75%
                frame_indices = [
                    int(0.25 * total_frames),
                    int(0.5 * total_frames),
                    int(0.75 * total_frames),
                ]

                # Loop through frames and extract frames at the specified indices
                saved_count = 0
                for frame_index in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()

                    if ret:
                        # Check if the frame has a face
                        if has_face(frame):
                            # Detect faces in the frame
                            detector = dlib.get_frontal_face_detector()
                            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = detector(gray_frame)

                            # Check if faces are detected
                            if faces:
                                # Crop and save the image for each detected face
                                for i, face in enumerate(faces):
                                    x, y, w, h = (
                                        face.left(),
                                        face.top(),
                                        face.width(),
                                        face.height(),
                                    )
                                    face_image = frame[y : y + h, x : x + w]

                                    # Check if face_image is not empty before saving
                                    if face_image is not None and face_image.size != 0:
                                        cropped_frame_filename = f"{output_folder}/{manipulated}_frame_{video_index}_{frame_index}_face_{i}.jpg"
                                        cv2.imwrite(cropped_frame_filename, face_image)
                                        saved_count += 1

                # Release the video capture object
                cap.release()

                print(f"Saved {saved_count} frames")

                # Increment the video index
                video_index += 1

    print("Frames extraction from all videos completed.")

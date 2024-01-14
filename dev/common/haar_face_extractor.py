import cv2
import os

def has_face(frame):
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

def extract_frames(input_folder, output_folder, faked=True):
    manipulated = "faked" if faked else "original"
    video_index = 0

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Count the total number of video files
    total_videos = 0
    if faked == True:
        total_videos = sum(1 for file_name in os.listdir(input_folder) if file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    else:
        total_videos = sum(1 for file_name in os.listdir(input_folder))

    # Loop through all files and subdirectories in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if it's a video file
            if file_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                print(f"Processing video {video_index + 1}/{total_videos}: {file_name} | ", end="")

                # Open the video file
                cap = cv2.VideoCapture(file_path)

                # Get the total number of frames in the video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Loop through frames and extract frames every 5 frames
                saved_count = 0
                for frame_index in range(0, total_frames, 5):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()

                    if ret:
                        # Check if the frame has a face
                        if has_face(frame):
                            # Detect faces in the frame
                            face_cascade = cv2.CascadeClassifier(
                                cv2.data.haarcascades
                                + "haarcascade_frontalface_default.xml"
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
                                cropped_frame_filename = f"{output_folder}/{manipulated}_frame_{video_index}_{frame_index}_face_{i}.jpg"
                                cv2.imwrite(cropped_frame_filename, face_image)

                            saved_count += 1

                # Release the video capture object
                cap.release()

                print(f"Saved {saved_count} frames")

                # Increment the video index
                video_index += 1

    print("Frames extraction from all videos completed.")


if __name__ == "__main__":
    extract_frames("./dev/common/temp", "./temp2", faked=True)

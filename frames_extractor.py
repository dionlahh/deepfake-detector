import cv2
import os


def extract_frames(video_path, output_folder, video_index, faked):
    """
    Extract frames from a video and save them as images at 25%, 50%, and 75% of the video.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_folder (str): Path to the folder where extracted frames will be saved.
    - faked (bool): True if the video is manipulated, False otherwise.

    Returns:
    - None
    """

    manipulated = "faked" if faked else "original"

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame indices for 25%, 50%, and 75%
    frame_indices = [int(0.25 * total_frames), int(0.5 * total_frames), int(0.75 * total_frames)]

    # Loop through frames and extract frames at the specified indices
    saved_count = 0
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            frame_filename = f"{output_folder}/{manipulated}_frame_{video_index}_{frame_index}.jpg"
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

    # Release the video capture object
    cap.release()

    print(f"Saved {saved_count} frames")


def extract_frames_from_folder(input_folder, output_folder, faked=True):
    """
    Extract frames from all videos in a folder and save them as images.

    Parameters:
    - input_folder (str): Path to the folder containing input video files or subfolders.
    - output_folder (str): Path to the folder where extracted frames will be saved.
    - frame_interval (int): Interval between frames to be extracted (default is 30 frames).

    Returns:
    - None
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files and subdirectories in the input folder
    video_index = 0
    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if it's a video file
            if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Processing video: {file_name} | ", end='')
                extract_frames(file_path, output_folder, video_index, faked)
                video_index += 1

    print("Frames extraction from all videos completed.")



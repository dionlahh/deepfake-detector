import cv2
import os

def extract_frames(input_folder, output_folder, faked=True, max_frames=None, max_frames_per_video=None):
    manipulated = "faked" if faked else "original"
    video_index = 0
    total_frames_extracted = 0

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .mp4 files in the input folder
    video_files = [file_name for file_name in os.listdir(input_folder) if file_name.lower().endswith(".mp4")]

    # Count the total number of video files
    total_videos = len(video_files)

    # Loop through .mp4 files in the input folder
    for file_name in video_files:
        file_path = os.path.join(input_folder, file_name)

        print(f"Processing video {video_index + 1}/{total_videos}: {file_name} | ", end="")

        # Open the video file
        cap = cv2.VideoCapture(file_path)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine the number of frames to extract based on max_frames_per_video
        frames_to_extract = range(0, total_frames, 5)
        if max_frames_per_video is not None and max_frames_per_video < len(frames_to_extract):
            frames_to_extract = frames_to_extract[:max_frames_per_video]

        # Loop through frames and extract frames every 5 frames
        saved_count = 0
        for frame_index in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret:
                # Save the frame
                frame_filename = f"{output_folder}/{manipulated}_frame_{video_index}_{frame_index}.jpg"
                cv2.imwrite(frame_filename, frame)

                saved_count += 1
                total_frames_extracted += 1

                # Stop if the maximum number of frames is reached
        if max_frames is not None and total_frames_extracted >= max_frames:
            break

        # Release the video capture object
        cap.release()

        print(f"Saved {saved_count} frames")

        # Increment the video index
        video_index += 1

    print("Frames extraction from all videos completed.")
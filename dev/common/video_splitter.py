# Used to split the videos into Train, Validation and Test sets

import os
import random
import shutil

def create_directory_structure(base_directory):
    # Define the directory names
    categories = ['Test', 'Training', 'Validation']
    subcategories = ['manipulated', 'original']
    subsubcategories = ['videos', 'frames', 'faces']

    # Create the base directory if it doesn't exist
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    created_paths = {}

    # Create the directory structure
    for category in categories:
        category_path = os.path.join(base_directory, category)
        os.makedirs(category_path)
        created_paths[category] = category_path
        
        for subcategory in subcategories:
            subcategory_path = os.path.join(category_path, subcategory)
            os.makedirs(subcategory_path)

            for subsubcategory in subsubcategories:
                subsubcategory_path = os.path.join(subcategory_path, subsubcategory)
                os.makedirs(subsubcategory_path)

    return created_paths


def random_select_and_move(source_dir, output_dir, percentage, not_selected_dir=None):
    # Get the list of files in the source directory
    files = os.listdir(source_dir)
    
    # Calculate the number of files to be selected based on the specified percentage
    num_files_to_select = int(len(files) * percentage)
    
    # Randomly select files
    selected_files = random.sample(files, num_files_to_select)
    
    # Ensure the output directory exists; create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Move selected files to the output directory
    for file in selected_files:
        source_path = os.path.join(source_dir, file)
        output_path = os.path.join(output_dir, file)
        shutil.move(source_path, output_path)

    # Move not selected files to the specified directory if provided
    if not_selected_dir is not None:
        not_selected_files = set(files) - set(selected_files)
        for file in not_selected_files:
            source_path = os.path.join(source_dir, file)
            not_selected_path = os.path.join(not_selected_dir, file)
            shutil.move(source_path, not_selected_path)


def split_data(base_directory, fake_dir, real_dir):
    created_paths = create_directory_structure(base_directory)
    train_dir = created_paths['Training']
    test_dir = created_paths['Test']
    val_dir = created_paths['Validation']

    MANIPULATED_VIDEOS = 'manipulated/videos'
    REAL_VIDEOS = 'original/videos'

    # Split train and test videos
    TRAIN_PERCENTAGE = 0.8
    random_select_and_move(fake_dir, os.path.join(train_dir, MANIPULATED_VIDEOS), TRAIN_PERCENTAGE, os.path.join(test_dir, MANIPULATED_VIDEOS))
    random_select_and_move(real_dir, os.path.join(train_dir, REAL_VIDEOS), TRAIN_PERCENTAGE, os.path.join(test_dir, REAL_VIDEOS))

    # Split train and validation videos
    VALIDATION_PERCENTAGE = 0.2
    random_select_and_move(os.path.join(train_dir, MANIPULATED_VIDEOS), os.path.join(val_dir, MANIPULATED_VIDEOS), VALIDATION_PERCENTAGE)
    random_select_and_move(os.path.join(train_dir, REAL_VIDEOS), os.path.join(val_dir, REAL_VIDEOS), VALIDATION_PERCENTAGE)


def clean_downloaded_videos(folder_path):
    # Get the list of subdirectories in the downloaded_videos folder
    subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(folder_path, subdirectory)
        
        # Look for .mp4 files in the subdirectory
        mp4_files = [file for file in os.listdir(subdirectory_path) if file.endswith('.mp4')]

        if mp4_files:
            # Move the .mp4 file to the parent directory (downloaded_videos)
            mp4_file = mp4_files[0]  # Assume there's only one .mp4 file per subdirectory
            source_path = os.path.join(subdirectory_path, mp4_file)
            destination_path = os.path.join(folder_path, mp4_file)
            shutil.move(source_path, destination_path)

            # Delete the rest of the contents in the subdirectory
            for file in os.listdir(subdirectory_path):
                file_path = os.path.join(subdirectory_path, file)
                if os.path.isfile(file_path) and file != mp4_file:
                    os.remove(file_path)

            # Remove the subdirectory itself along with its contents
            shutil.rmtree(subdirectory_path)


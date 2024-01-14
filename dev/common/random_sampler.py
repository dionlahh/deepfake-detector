import os
import random
import shutil

def copy_random_files(input_dir, output_dir, x):
    # Get the list of files in the input directory
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Ensure x is not greater than the number of files available
    x = min(x, len(files))

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Randomly select x files
    selected_files = random.sample(files, x)

    # Copy selected files to the output directory
    for i, file_name in enumerate(selected_files):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        shutil.copyfile(input_path, output_path)

        # Print completion percentage
        completion_percentage = ((i + 1) / x) * 100
        print(f"Copying: {completion_percentage:.2f}% complete", end='\r')

    print("\nCopy completed.")

def delete_non_selected_files(input_dir, x):
    # Get the list of files in the input directory
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Ensure x is not greater than the number of files available
    if x >= len(files):
        print("Number provided is larger than the number of files available!")
        return    

    # Randomly select x files
    selected_files = random.sample(files, x)

    # Delete non-selected files
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        if file_name not in selected_files:
            os.remove(file_path)

    print("Deletion completed.")


if __name__ == "__main__":
    # copy_random_files("./dev/train1/manipulated/frames", "./dev/traintozip/manipulated/frames", 40000)
    # copy_random_files("./dev/train1/original/frames", "./dev/traintozip/original/frames", 40000)
    delete_non_selected_files("./dev/Split Data/Test/original/frames", 40000)
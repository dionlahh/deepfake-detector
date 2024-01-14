import os
import dlib
import numpy as np
from PIL import Image
import random
import shutil
import zipfile
import fnmatch
import matplotlib



def delete_images_without_faces(folder_path):
    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Get the total number of images in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
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
            print(f"No face detected in {filename}. Deleting the image.")
            os.remove(file_path)

        # Update progress
        processed_images += 1
        progress_percentage = (processed_images / total_images) * 100
        print(f"Progress: {processed_images}/{total_images} images processed ({progress_percentage:.2f}%)")


def copy_random_images(input_path, output_path, num_images_to_copy):
    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Check if there are enough images to copy
    if len(image_files) < num_images_to_copy:
        print(f"Error: There are not enough images in the input path.")
        return

    # Randomly select num_images_to_copy unique images
    selected_images = random.sample(image_files, num_images_to_copy)

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Copy the selected images to the output directory
    for image in selected_images:
        input_image_path = os.path.join(input_path, image)
        output_image_path = os.path.join(output_path, image)

        shutil.copy(input_image_path, output_image_path)
        print(f"Image '{image}' copied to '{output_path}'")

def zip_directory(input_path, output_zip):
    # Ensure that the input path exists
    if not os.path.exists(input_path):
        print(f"Error: The specified directory '{input_path}' does not exist.")
        return

    # Create a zip file from the directory
    shutil.make_archive(output_zip, 'zip', input_path)

    print(f"Directory '{input_path}' successfully zipped to '{output_zip}.zip'")

def create_data_zip(no_of_images):
    faked = "dev/data/frames/Haar_Cascade/faked"
    output_faked = "dev/data/frames/Haar_Cascade/training_data/manipulated"
    original = "dev/data/frames/Haar_Cascade/original"
    output_original = "dev/data/frames/Haar_Cascade/training_data/original"
    dir_to_zip = "dev/data/frames/Haar_Cascade/training_data"
    zip_output = "dev/data"

    copy_random_images(faked, output_faked, no_of_images)
    copy_random_images(original, output_original, no_of_images)

    zip_directory(dir_to_zip, zip_output)

# Function which assigns the file to the correct directory based on the discrete cumulative distribution function cdf
def assign_data(cdf):
    nbr=random.random()
    for set in list(cdf.keys()):
        if nbr<cdf[set]:
            return set

def split_train_val():
    # Split Data
    # ├── Test  
    # │   ├── manipulated
    # │   └── original
    # ├── Training
    # │   ├── manipulated 
    # │   └── original
    # └── Validation
    #     ├── manipulated 
    #     └── original

    # Check if the dataset directory is already present to avoid redundant read and writes
    isExist = os.path.exists('./Split Data/')
    
    if not isExist:
    # Creates the appropriate directory structures for training, validation and test sets.
        try:
            shutil.rmtree('./Split Data')         
        except:
            pass                #Split Data didn't exist
        
        os.mkdir('./Split Data')
        cdf={"Training":0.7,"Validation":0.85,"Test":1} #OBS! Has to be increment percentages of 5 to make batch size fit
        for dir in list(cdf.keys()):
            os.mkdir('./Split Data/{}'.format(dir))
            os.mkdir('./Split Data/{}/manipulated'.format(dir))
            os.mkdir('./Split Data/{}/original'.format(dir))

        if not isExist:
            dist={"Training":0,"Validation":0,"Test":0}
            filename = 'data.zip'
            zfile = zipfile.ZipFile(filename, 'r')
            counter=0
            manipulated_counter, original_counter = 0, 0
            samplesize=2321
            # Each file is loaded in sequence and randomly assigned to the corresponding directory 
            # in the new straucture according to cdf. Dictionary dist keeps track of number of each set.
            for name in zfile.namelist():
                save_path = './Split Data/'
                name_of_file=""
                label=""
                if fnmatch.fnmatch(name, "data/manipulated/*.jpg") and manipulated_counter < samplesize:
                    name_of_file=name[len("data/manipulated/"):]
                    label="manipulated"
                    manipulated_counter += 1
                if fnmatch.fnmatch(name,"data/original/*.jpg") and original_counter < samplesize:
                    name_of_file=name[len("data/original/"):]
                    label="original"
                    original_counter += 1
                if name_of_file != "":
                    myfile = zfile.open(name)
                    img = matplotlib.image.imread(myfile)
                    rand_assign=assign_data(cdf)
                    dist[rand_assign]+=1
                    save_path+=rand_assign+"/"+label # eg. "Split Data/Training/manipulated
                    completeName = os.path.join(save_path, name_of_file)         
                    matplotlib.image.imsave(completeName,img)
                    counter+=1
                    if manipulated_counter == original_counter and manipulated_counter == samplesize:     
                        break

            print(counter)

            zfile.close()

if __name__ == '__main__':
    create_data_zip(no_of_images=80000)



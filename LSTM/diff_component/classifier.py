import os
import shutil
import glob

# Define the directory where the images are located
image_dir = '/Users/rhine_e/Downloads/CrossPatternData/training-set-backup'

# Define the directory where the sorted images will be stored
output_dir = './sorted_images'

A, B, C, D, E, F, G, H, I, J, K, L = (0,0.5,0.5),(0.125,0.5,0.375),(0.25,0.5,0.25),(0.375,0.5,0.125),(0.5,0.5,0),(0.5,0.375,0.125),(0.5,0.25,0.25),(0.5,0.125,0.375),(0.5,0,0.5),(0.375,0.125,0.5),(0.25,0.25,0.5),(0.125,0.375,0.5)

# Define the specific mass proportions
mass_proportions = [A,L,K,J,I]

# Get a list of all the images in the directory
images = glob.glob(os.path.join(image_dir, '*.png'))

# Initialize a counter for the subfolders
subfolder_counter = 0

# Sort the images by temperature, pressure, and cycle number
images.sort(key=lambda x: (int(os.path.basename(x).split('-')[3]), int(os.path.basename(x).split('-')[4]), int(os.path.basename(x).split('-')[5])))

# Initialize variables to track the current group
current_temp = current_pressure = current_cycle = None

for image in images:
    # Split the image name into its components
    components = os.path.basename(image).split('-')
    # print (os.path.basename(image).split('-')[1:6])

    # Check if the frame number is 11
    if int(components[6].split('.')[0]) == 11:
        # Check if the mass proportions match one of the specified groups
        mass_prop = (float(components[0]), float(components[1]), float(components[2]))
        if mass_prop in mass_proportions:
            # Check if the temperature, pressure, and cycle number have changed
            if (int(components[3]), int(components[4]), int(components[5])) != (current_temp, current_pressure, current_cycle):
                # Update the current group
                current_temp, current_pressure, current_cycle = int(components[3]), int(components[4]), int(components[5])

                # Increment the subfolder counter
                subfolder_counter += 1

            # Create a new subfolder for this group of images
            new_subfolder = os.path.join(output_dir, str(subfolder_counter))
            os.makedirs(new_subfolder, exist_ok=True)

            # Move the image to the new subfolder and rename it
            filename, extension = os.path.splitext(os.path.basename(image))
            if mass_prop == mass_proportions[0]: filename = '1'
            if mass_prop == mass_proportions[1]: filename = '2'
            if mass_prop == mass_proportions[2]: filename = '3'
            if mass_prop == mass_proportions[3]: filename = '4'
            if mass_prop == mass_proportions[4]: filename = '5'
            new_image_name = filename + extension
            shutil.copy(image, os.path.join(new_subfolder, new_image_name))

# After sorting and moving all images, check each subfolder
for folder in os.listdir(output_dir):
    folder_path = os.path.join(output_dir, folder)

    # Ensure the path is indeed a folder
    if os.path.isdir(folder_path):
        # Count the number of images in the folder
        num_images = len(glob.glob(os.path.join(folder_path, '*.png')))
        print('Number of images in folder {}: {}'.format(folder, num_images))
        # If the number of images is less than 5, delete the folder
        if num_images < 5:
            shutil.rmtree(folder_path)

# Create a mapping of old folder names to new folder names
folders = sorted([int(folder) for folder in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, folder))])
folder_mapping = {str(old): str(new) for new, old in enumerate(folders, start=1)}

# Rename the folders according to the mapping
for old, new in folder_mapping.items():
    old_folder_path = os.path.join(output_dir, old)
    new_folder_path = os.path.join(output_dir, new)

    # Check if the new folder path exists
    if not os.path.exists(new_folder_path):
        os.rename(old_folder_path, new_folder_path)
    else:
        print(f"Cannot rename {old_folder_path} to {new_folder_path} because the latter already exists.")
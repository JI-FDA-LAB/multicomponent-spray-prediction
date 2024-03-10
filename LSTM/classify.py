import os
import shutil

# The directory where your images are currently stored
src_dir = '/Users/rhine_e/Downloads/CrossPatternData/training-set-backup'

# The directory where you want to move your images
dst_dir = '/Users/rhine_e/Downloads/CrossPatternData/LSTM/training_data'

# Function to sort filenames based on each part of the filename
def sort_key(filename):
    # Split the filename from its extension
    name, _ = os.path.splitext(filename)
    # Split the filename on '-' and convert each part to an integer
    parts = map(float, name.split('-'))
    # Return the parts as a tuple
    return tuple(parts)

# Get a sorted list of all image files
img_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')], key=sort_key)
# print(img_files)

# Initialize folder count
folder_count = 1

# Iterate over the images in groups of 16
for i in range(0, len(img_files), 16):
    # Check if the sequence is complete
    if i + 16 <= len(img_files) and all(str(j+1) in img_files[i+j] for j in range(16)):
        # Create a new directory for this group of images
        new_dir = os.path.join(dst_dir, str(folder_count))
        os.makedirs(new_dir, exist_ok=True)

        # Move the images to the new directory and rename them
        for j in range(i, i + 16):
            # Construct the new image name by removing the '-' characters
            filename, extension = os.path.splitext(img_files[j])
            new_img_name = filename.replace('-', '').replace('.','') + extension

            # Construct the source and destination paths
            src_path = os.path.join(src_dir, img_files[j])
            dst_path = os.path.join(new_dir, new_img_name)

            # Move and rename the image
            shutil.copy(src_path, dst_path)

    # Increment the folder count
    folder_count += 1
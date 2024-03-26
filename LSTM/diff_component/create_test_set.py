import os
import shutil

# Define the source directory and target directories
src_dir = '../../../training-set'  # Replace with your actual directory
train_dir = './training-set'  # Replace with your desired directory
test_dir = './test-set'  # Replace with your desired directory

# Create the target directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate over the files in the source directory
for filename in os.listdir(src_dir):
    # Check if the filename follows the expected format
    parts = filename.split('-')
    if len(parts) < 6:
        print(f'Skipping file {filename} as it does not follow the expected format')
        continue

    # Parse the cycle number from the filename
    cycle_number = int(parts[5])  # Adjust based on your actual filename format

    # Determine the target directory based on the cycle number
    if 17 <= cycle_number <= 20:
        target_dir = test_dir
    else:
        target_dir = train_dir

    # Move the file to the target directory
    shutil.copy(os.path.join(src_dir, filename), os.path.join(target_dir, filename))

import cv2
import os
import numpy as np

MASK = 0.1

# Preprocess images by resizing them from (768,768) to (512,512)
def preprocess_image(image):
    # Resize the image to (512, 512)
    resized_image = cv2.resize(image, (512, 512))
    return resized_image

# Calculate the plume-to-tip distance (d) of a spray pattern image
def calculate_d(image):
    max_intensity = image.max()
    threshold_value = int(max_intensity * MASK)
    _, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Preserve intensity information
    thresholded_image = cv2.bitwise_and(image, binary_mask)

    # Calculate plume-to-tip distance (d)
    height, width = image.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    injector_tip_position = (235, 243)  # Tip coordinate adjusted for resized image
    distances_to_tip = np.sqrt((x_coords - injector_tip_position[0])**2 + (y_coords - injector_tip_position[1])**2)
    d = np.sum(thresholded_image * distances_to_tip) / np.sum(thresholded_image)

    return d

# Calculate the relative error between the input images and the label image
# The commented line is for extrapolation case
def calculate_parameters(input_path, num_images=3):
# def calculate_parameters(input_path, num_images=5):
    errors = [[] for _ in range(num_images - 1)]

    with open(os.path.join(input_path, "d.txt"), "w") as file:
        subfolders = sorted(os.listdir(input_path))
        for subfolder in subfolders:
            subfolder_path = os.path.join(input_path, subfolder)
            if os.path.isdir(subfolder_path):
                images = [cv2.imread(os.path.join(subfolder_path, f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(1, num_images + 1)]
                preprocessed_images = [preprocess_image(image) for image in images]
                d_values = [calculate_d(image) for image in preprocessed_images]

                for i in range(num_images - 1):
                    relative_error = abs(d_values[i] - d_values[-1]) / d_values[-1]
                    errors[i].append(relative_error)
                    file.write(f"{subfolder} Relative Error{i + 1}: {relative_error}\n")

    mean_relative_errors = [np.mean(error_list) for error_list in errors]

    for i in range(len(mean_relative_errors)):
        print(f"Similarity for {i + 1} & {num_images}: {100 - mean_relative_errors[i] * 100}%")


image_path = './dataset/sample_test/interpolation/AEC-test-set'
# image_path = './dataset/sample_test/extrapolation/ABCDE-test-set'
calculate_parameters(image_path)
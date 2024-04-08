import os
import cv2
import numpy as np

# Function to apply threshold and separate contour from background
def apply_threshold_and_separate_contour(input_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    overall_centroid_x = 0
    overall_centroid_y = 0
    total_images = 0

    # Iterate through each image in the input directory
    for filename in os.listdir(input_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            total_images += 1

            # Read image
            img = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)

            # Calculate 10% of the maximum intensity value
            max_intensity = img.max()
            min_intensity = img.min()
            print (filename + " max intensity is: {}, min intensity is: {}.".format(max_intensity, min_intensity))
            threshold_value = int(max_intensity * 0.1)

            # Apply threshold
            _, binary_mask = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

            thresholded_image = cv2.bitwise_and(img, binary_mask)

            # Save image with overall centroid marker
            cv2.imwrite(os.path.join(output_path, filename), thresholded_image)


# Example usage
input_dir = '/Users/rhine_e/Downloads/CrossPatternData/test-set'
output_dir = '/Users/rhine_e/Downloads/CrossPatternData/thresholded-test-set'

apply_threshold_and_separate_contour(input_dir, output_dir)

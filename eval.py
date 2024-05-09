import cv2
import os
import numpy as np

MASK = 0.1

def preprocess_image(image):
    # Resize the image to (512, 512)
    resized_image = cv2.resize(image, (512, 512))
    return resized_image

def calculate_parameters(input_path):
    d1_values = []
    d2_values = []
    d3_values = []
    d4_values = []
    with open(os.path.join(input_path, "d.txt"), "w") as file:
        subfolders = sorted(os.listdir(input_path))
        for subfolder in subfolders:
            subfolder_path = os.path.join(input_path, subfolder)
            if os.path.isdir(subfolder_path):
                # Load images "1", "2", and "3"
                image1 = cv2.imread(os.path.join(subfolder_path, "1.png"), cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(os.path.join(subfolder_path, "2.png"), cv2.IMREAD_GRAYSCALE)
                image3 = cv2.imread(os.path.join(subfolder_path, "3.png"), cv2.IMREAD_GRAYSCALE)
                # image4 = cv2.imread(os.path.join(subfolder_path, "4.png"), cv2.IMREAD_GRAYSCALE)
                # image5 = cv2.imread(os.path.join(subfolder_path, "5.png"), cv2.IMREAD_GRAYSCALE)

                if image1 is None or image2 is None or image3 is None:
                    print(f"Error loading images in subfolder: {subfolder}")
                    continue

                # Preprocess images
                image1 = preprocess_image(image1)
                image2 = preprocess_image(image2)
                image3 = preprocess_image(image3)
                # image4 = preprocess_image(image4)
                # image5 = preprocess_image(image5)


                # Calculate 'd' for images "1", "2", and "3"
                d1, thresholded_image1, img_line1 = calculate_d(image1)
                d2, thresholded_image2, img_line2 = calculate_d(image2)
                d3, thresholded_image3, img_line3 = calculate_d(image3)
                # d4, thresholded_image4, img_line4 = calculate_d(image4)
                # d5, thresholded_image5, img_line5 = calculate_d(image5)

                # cv2.imwrite(os.path.join(subfolder_path, f"2_processed.png"), thresholded_image2)
                # Save or display the image with the line
                # cv2.imwrite(os.path.join(subfolder_path, f"{1}_with_line.png"), img_line1)
                # cv2.imwrite(os.path.join(subfolder_path, f"{2}_with_line.png"), img_line2)
                # cv2.imwrite(os.path.join(subfolder_path, f"{3}_with_line.png"), img_line3)
                # cv2.imwrite(os.path.join(subfolder_path, f"{4}_with_line.png"), img_line4)
                # cv2.imwrite(os.path.join(subfolder_path, f"{5}_with_line.png"), img_line5)

                # Calculate relative error
                relative_error1 = abs(d1 - d3) / d3
                relative_error2 = abs(d2 - d3) / d3
                # relative_error3 = abs(d3 - d5) / d5
                # relative_error4 = abs(d4 - d5) / d5

                d1_values.append(relative_error1)
                d2_values.append(relative_error2)
                # d3_values.append(relative_error3)
                # d4_values.append(relative_error4)

                # Write to file
                file.write(f"{subfolder} Relative Error1: {relative_error1}\n")
                file.write(f"{subfolder} Relative Error2: {relative_error2}\n")
                # file.write(f"{subfolder} Relative Error3: {relative_error3}\n")
                # file.write(f"{subfolder} Relative Error4: {relative_error4}\n")

    # Calculate and print overall relative error
    mean_relative_error1 = np.mean(d1_values)
    mean_relative_error2 = np.mean(d2_values)
    # mean_relative_error3 = np.mean(d3_values)
    # mean_relative_error4 = np.mean(d4_values)
    print("Overall Mean Similarity for 1 & 3:", 100 - mean_relative_error1 * 100, "%")
    print("Overall Mean Similarity for 2 & 3:", 100 - mean_relative_error2 * 100, "%")
    # print("Overall Mean Similarity for 3 & 5:", 100 - mean_relative_error3 * 100, "%")
    # print("Overall Mean Similarity for 4 & 5:", 100 - mean_relative_error4 * 100, "%")

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

    d_x = d * 0.788
    d_y = -d * 0.616
    line_x = int(round(d_x))
    line_y = int(round(d_y))
    # line_length = int(round(d))
    image_with_line = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)  # Convert image to color (BGR) format
    cv2.line(image_with_line, (injector_tip_position[0], injector_tip_position[1]), (injector_tip_position[0] + line_x, injector_tip_position[1] + line_y), (0, 0, 255), 2)  # Draw red line
    cv2.circle(image_with_line, (injector_tip_position[0], injector_tip_position[1]), 5, (0, 255, 255), -1)  # Draw yellwo dot
    cv2.circle(image_with_line, (injector_tip_position[0] + line_x, injector_tip_position[1] + line_y), 5, (0, 255, 255), -1)

    return d, thresholded_image, image_with_line

# Function to read lines from a text file
def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

# image_path = './AEC-test-set-threshold'
image_path = './KIJ-test-set'
calculate_parameters(image_path)
import cv2
import os
import numpy as np

def calculate_parameters(input_path):
    total_images = 0
    with open(os.path.join(input_path, "d.txt"), "w") as file:
        for filename in os.listdir(input_path):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                total_images += 1
                # Thresholding to distinguish spray from background
                image = cv2.imread(os.path.join(input_path, filename), cv2.IMREAD_GRAYSCALE)
                # if filename.startswith('output'):
                # image = image * 2.0 - 255.0
                # image = np.clip(image, 0, 255).astype(np.uint8)
                # if image is None:
                #     print(f"Error loading image: {os.path.join(input_path, filename)}")
                #     continue
                max_intensity = image.max()
                min_intensity = image.min()
                print(filename + " max intensity is: {}, min intensity is: {}.\n".format(max_intensity, min_intensity))
                # if filename.startswith('output'):
                #     threshold_value = int(max_intensity * 0.2)
                # else:
                #     threshold_value = int(max_intensity * 0.2)
                # if max_intensity < 120:
                #     threshold_value = int(max_intensity * 0.4)
                # else:
                #     threshold_value = int(max_intensity * 0.7)
                threshold_value = int(max_intensity * 0.2)
                _, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

                # Preserve intensity information
                thresholded_image = cv2.bitwise_and(image, binary_mask)

                # Calculate plume-to-tip distance (d)
                height, width = image.shape
                y_coords, x_coords = np.mgrid[0:height, 0:width]
                injector_tip_position = (235, 243)  # Tip coordinate
                distances_to_tip = np.sqrt((x_coords - injector_tip_position[0])**2 + (y_coords - injector_tip_position[1])**2)
                d = np.sum(thresholded_image * distances_to_tip) / np.sum(thresholded_image)

                # Calculate spray cross-pattern area (Ac)
                contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                total_area = sum(cv2.contourArea(contour) for contour in contours)

                # Visualize plume-to-tip distance (d) with a horizontal line
                image_with_line = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)  # Convert image to color (BGR) format
                line_length = int(round(d))  # Round the distance to the nearest integer
                cv2.line(image_with_line, (injector_tip_position[0], injector_tip_position[1]), (injector_tip_position[0] + line_length, injector_tip_position[1]), (0, 0, 255), 1)  # Draw red line

                # Save or display the image with the line
                cv2.imwrite(os.path.join(input_path, f"{filename}_with_line.png"), image_with_line)

                file.write(f"{filename} Plume-to-tip distance (d): {d}\n")
                # file.write(f"{filename} Spray cross-pattern area (Ac): {total_area}\n")
    # print(f"Total images processed: {total_images}")

# Function to read lines from a text file
def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

# Define a custom sorting function to sort lines based on the number in the PNG filename
def sort_key(line):
    # Extract the number from the PNG filename
    filename = line.split()[0]  # Get the first part of the line (the filename)
    number = int(filename.split('_')[-1].split('.')[0])  # Extract the number from the filename
    return number


# image_path = './3-group-test-results/EIG_0.0001_500_threshold/test_results'
image_path = './3-group-test-results/EGF_predict_threshold'
calculate_parameters(image_path)

lines = read_lines_from_file(os.path.join(image_path,'d.txt'))

# Sort the lines using the custom sorting function
sorted_lines = sorted(lines, key=sort_key)

file_path = os.path.join(image_path, 'sorted_d.txt')

# Write the sorted lines back to a new text file
with open(file_path, 'w') as file:
    file.writelines(sorted_lines)

output_values = []
label_values = []

# Open the file and read line by line
with open(file_path, 'r') as file:
    for line in file:
        parts = line.split()
        d_value = float(parts[-1])
        file_name = parts[0]
        # Append the value to the appropriate list based on the file name
        if 'output' in file_name:
            output_values.append(d_value)
        elif 'label' in file_name:
            label_values.append(d_value)

# Calculate the relative error for each pair of values
relative_errors = np.abs(np.array(output_values) - np.array(label_values)) / np.abs(np.array(label_values))
mean_relative_error = np.mean(relative_errors)

print("Mean Relative Error:", mean_relative_error * 100, "%")
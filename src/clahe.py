""" 
This script applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance underwater images. 
It reads images from the specified input folder, processes them using CLAHE, and saves the enhanced images to the output folder. 
The script also prints progress updates for each processed image.
"""

import cv2
import os

# Input & Output folders
input_folder = "data/raw_subset"
output_folder = "data/clahe_output"

os.makedirs(output_folder, exist_ok=True)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):

        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE on L channel
        cl = clahe.apply(l)

        # Merge back
        merged = cv2.merge((cl, a, b))

        # Convert back to BGR
        final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Save result
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, final)

        print(f"Processed: {filename}")

print("Completed")

# this is to create side by side before and after comparsions of the images
#idk if this is need, but yeah we have it.

import cv2
import os
import numpy as np

input_folder = "../data/raw"
clahe_folder = "../data/clahe"
output_folder = "../results/images"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):

        original = cv2.imread(os.path.join(input_folder, filename))
        clahe = cv2.imread(os.path.join(clahe_folder, filename))

        combined = np.hstack((original, clahe))

        cv2.imwrite(os.path.join(output_folder, filename), combined)

print("Comparison images created")
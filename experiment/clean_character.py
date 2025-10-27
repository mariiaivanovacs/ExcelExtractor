import os 
import pandas as pd 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')



def mask_pixels(img, output_path="masked_image.png"):
    import numpy as np
    import cv2

    # Convert to float 0–1
    img = img.astype(np.float32) / 255.0

    # If image is grayscale (2D), expand to 3D so we can use axis=2
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)

    # Compute brightness and color variation
    brightness = np.mean(img, axis=2)
    color_diff = np.max(img, axis=2) - np.min(img, axis=2)

    # Threshold
    bright_thresh = 0.7
    color_thresh = 0.1

    # Mask for light gray pixels
    mask = (brightness > bright_thresh) & (color_diff < color_thresh)

    # Replace with white
    img[mask] = [1.0, 1.0, 1.0]

    # Convert back to uint8
    result = (img * 255).astype(np.uint8)
    
    
    # cv2.imwrite(f"steps_out/{output_path}", result)
    return result


directory = "tests"
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        file_path = os.path.join(directory, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = mask_pixels(img)
        cv2.imwrite(file_path, img)

# file_path = "tests/cell_r20_c18_blob_1_word_3_char_01.png"

# img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# img = mask_pixels(img)

# # store for debug
# cv2.imwrite(file_path, img)

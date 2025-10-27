import numpy as np
import cv2
from typing import Tuple, Optional

def largest_black_region_size(np_img: np.ndarray, min_area: int = 1) -> Tuple[int, int]:
    """
    Find the largest connected black region (character/shape) in `np_img`.
    Returns (height, width) of its bounding box in pixels.
    If no region found (or all are smaller than min_area) returns (0, 0).

    Parameters
    ----------
    np_img : np.ndarray
        Input image. Can be grayscale (H,W) or color (H,W,3). dtype should be uint8 or convertible.
    min_area : int
        Minimum pixel area for a region to be considered (default 1).

    Returns
    -------
    (height, width) : Tuple[int, int]
        Height and width of the bounding box around the largest black region.
    """
    if np_img is None:
        return (0, 0)

    # Ensure uint8
    img = np_img.copy()
    if img.dtype != np.uint8:
        img = (255 * (img.astype('float32') / np.max(img))).astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)

    # Convert color -> grayscale if needed
    if img.ndim == 3 and img.shape[2] in (3, 4):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Otsu thresholding but invert so that black content (low intensity) becomes foreground (255)
    # This handles normal black-on-white images robustly.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Remove tiny specks (optional): morphological opening can be added if needed.
    # connectedComponentsWithStats expects 0-background, non-zero foreground.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    # stats: rows for labels; columns: [cv2.CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA]
    # skip label 0 (background)
    if num_labels <= 1:
        return (0, 0)

    # Find largest component by area that is >= min_area
    max_area = 0
    max_label = -1
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area > max_area:
            max_area = area
            max_label = label

    if max_label == -1:
        return (0, 0)

    w = int(stats[max_label, cv2.CC_STAT_WIDTH])
    h = int(stats[max_label, cv2.CC_STAT_HEIGHT])
    return (h, w)


# if __name__ == "__main__":
#     import numpy as np
#     import cv2

#     # create a white canvas
#     # img = 255 * np.ones((200, 300), dtype=np.uint8)

#     # # draw two black rectangles (simulating characters); one larger than the other
#     # cv2.rectangle(img, (20, 30), (80, 150), 0, -1)   # smaller
#     # cv2.rectangle(img, (150, 40), (270, 170), 0, -1) # larger
    
#     img = cv2.imread("characters/cell_r8_c18_blob_1_word_2_char_00.png", cv2.IMREAD_GRAYSCALE)

#     h, w = largest_black_region_size(img)
#     print("Largest black region size (height, width):", (h, w))
#     # Expected -> roughly (131, 121) depending on coordinates used above


import cv2
import numpy as np
from typing import Tuple, Optional

def preprocess_for_components(gray):
    # Threshold, invert (black content becomes white)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing fills tiny gaps or diagonal disconnections
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    return bw

def find_largest_black_bbox(np_img: np.ndarray, min_area: int = 1) -> Optional[Tuple[int,int,int,int]]:
    """
    Returns (x, y, w, h) of largest black connected component in image,
    or None if none found.
    """
    if np_img is None:
        return None

    img = np_img.copy()
    # ensure uint8
    if img.dtype != np.uint8:
        img = (255 * (img.astype('float32') / np.max(img))).astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)

    # grayscale
    if img.ndim == 3 and img.shape[2] in (3, 4):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold so black content becomes foreground (255)
    bw = preprocess_for_components(gray)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels <= 1:
        return None

    max_area = 0
    max_label = -1
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area and area > max_area:
            max_area = area
            max_label = lab

    if max_label == -1:
        return None

    x = int(stats[max_label, cv2.CC_STAT_LEFT])
    y = int(stats[max_label, cv2.CC_STAT_TOP])
    w = int(stats[max_label, cv2.CC_STAT_WIDTH])
    h = int(stats[max_label, cv2.CC_STAT_HEIGHT])
    return (x, y, w, h)


def center_crop_or_pad_horiz(img: np.ndarray, target_w: int, pad_side: str = 'left') -> np.ndarray:
    """
    For a grayscale img with shape (h, w):
     - if w >= target_w: center-crop horizontally to target_w.
     - if w < target_w: pad with white (255) on one side only.
       pad_side == 'left' -> pad on left side (new pixels on left)
       pad_side == 'right' -> pad on right side (new pixels on right)

    Returns image with shape (h, target_w).
    """
    target_w, target_h = 32, 32
    h, w = img.shape[:2]
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)
    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    # pad_top = 0
    # pad_bottom = 0
    
    dtype = img.dtype

    # --- Normalize background color based on dtype ---
    if np.issubdtype(dtype, np.floating):
        # floats in [0,1]
        white_val = 1.0
    else:
        white_val = 255

    if img.ndim == 3:
        bg = (white_val, white_val, white_val)
    else:
        bg = white_val
        
    padded = cv2.copyMakeBorder(
        img,
        top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=bg
    )

    return padded


def split_character_by_center_and_pad(
    np_img: np.ndarray,
    target_width: int = 32,
    ratio_threshold: float = 1,
    triple_ratio_threshold: float = 1.8,
    filename: str = "photo.png"
) -> Tuple:

    """
    Detect largest black character, check bounding box ratio w/h.
    - If ratio <= ratio_threshold: return (None, None, None, bbox)
    - If ratio_threshold < ratio <= triple_ratio_threshold: split into 2 parts
    - If ratio > triple_ratio_threshold: split into 3 parts
    Each part padded/cropped horizontally to target_width.
    """
    bbox = find_largest_black_bbox(np_img)
    if bbox is None:
        return None, None, None, None

    x, y, w, h = bbox
    ratio = w / float(h) if h != 0 else 0.0

    # ensure grayscale for cropping/padding
    img = np_img.copy()
    if img.dtype != np.uint8:
        img = (255 * (img.astype('float32') / np.max(img))).astype(np.uint8) if img.max() > 1 else (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] in (3,4):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- CASE 1: no split ---
    if ratio <= ratio_threshold:
        return None, None

    # --- CASE 2: split into two parts ---
    elif ratio <= triple_ratio_threshold:
        mid = x + w // 2
        left_crop = gray[y:y+h, x:mid].copy()
        right_crop = gray[y:y+h, mid:x+w].copy()

        left_out = center_crop_or_pad_horiz(left_crop, target_width, pad_side='right')
        right_out = center_crop_or_pad_horiz(right_crop, target_width, pad_side='left')
        
           
        # save them
        cv2.imwrite(f"characters/{filename}_1.png", left_out)
        cv2.imwrite(f"characters/{filename}_2.png", right_out)
        print("Saved 2 parts for {filename}")

        return left_out, right_out, None, bbox

    # --- CASE 3: split into three parts ---
    else:
        # Split width into 3 equal parts
        step = w // 3
        x1 = x
        x2 = x + step
        x3 = x + 2 * step
        x4 = x + w

        part1 = gray[y:y+h, x1:x2].copy()
        part2 = gray[y:y+h, x2:x3].copy()
        part3 = gray[y:y+h, x3:x4].copy()

        # Apply padding: left, middle, right
        part1_out = center_crop_or_pad_horiz(part1, target_width, pad_side='right')
        part2_out = center_crop_or_pad_horiz(part2, target_width, pad_side='right')
        part3_out = center_crop_or_pad_horiz(part3, target_width, pad_side='left')
        
        # save them
        cv2.imwrite(f"characters/{filename}_1.png", part1_out)
        cv2.imwrite(f"characters/{filename}_2.png", part2_out)
        cv2.imwrite(f"characters/{filename}_3.png", part3_out)
        print("Saved 3 parts for {filename}")

        return part1_out, part2_out, part3_out, bbox

# Example usage:
# # # ---------------------------
# if __name__ == "__main__":
#     # Example image: white background with one wide black rectangle (simulating a wide character)
#     # img = 255 * np.ones((64, 80), dtype=np.uint8)
#     # cv2.rectangle(img, (10, 10), (70, 54), 0, -1)  # a wide black shape

#     img = cv2.imread("characters/cell_r6_c14_blob_1_word_1_char_02.png", cv2.IMREAD_GRAYSCALE)
    
#     h, w = largest_black_region_size(img)
#     print(h, w)
#     print(w/h)

#     # # detect bbox and possibly split+pad
#     result = split_character_by_center_and_pad(img, target_width=32, ratio_threshold=1)
    


    # print("Detected bbox:", bbox)
    # if left_img is not None and right_img is not None:
    #     print("Left shape:", left_img.shape, " Right shape:", right_img.shape)
    #     # show with cv2.imshow if you want (requires GUI)
    #     # cv2.imshow("left", left_img); cv2.imshow("right", right_img); cv2.waitKey(0)
    # else:
    #     print("Character not wide enough to split (ratio <= 0.9).")


def check_size(img, filename):
    
    h, w = largest_black_region_size(img)
    if h < 7 and w < 7:
        return False
    if  w/h  > 1:
        print("Two characters")
        result = split_character_by_center_and_pad(img, target_width=32, ratio_threshold=1, filename=filename)
        if len(result) == 2:
            return False
        else:
            return True
    else:
        print("IT IS ONE")
        return False
        
        
import os
# import os 

# replaced_files = []      
# working_directory = "characters"
# for filename in os.listdir(working_directory):
#     if filename.endswith('.png'):
#         img_path = os.path.join(working_directory, filename)
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         result = check_size(img)
#         if not result:
#             pass
#         else:
#             # replace this character image with 2 others
#             # first_file = img_path.replace(".png", "_1.png")
#             # second_file = img_path.replace(".png", "_2.png")
#             # first_file = f"characters/{filename}_1.png"
#             # second_file = f"characters/{filename}_2.png"
#             # cv2.imwrite(first_file, result[0])
#             # cv2.imwrite(second_file, result[1])
#             # remove previos 
#             os.remove(img_path)
#             replaced_files.append(filename)

# print("Count of replaced files: ", len(replaced_files))
# print("Replaced files: ", replaced_files)

import pandas as pd   
replaced_files = []
# Load CSV file
df = pd.read_csv("data/csv/numbers.csv")

# Assuming your CSV has a column that contains filenames (e.g., "filename")
# You can adjust this column name if it’s different.
csv_filenames = set(df["filename"].astype(str))  # make lookup fast with a set

working_directory = "characters"

for filename in os.listdir(working_directory):
    if not filename.endswith('.png'):
        continue  # skip non-png files

    # Check if this filename is listed in the CSV
    if filename not in csv_filenames:
        continue  # skip files not in the CSV

    img_path = os.path.join(working_directory, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(f"Processing file: {filename}")

    result = check_size(img, filename)
    if result:
        replaced_files.append(filename)
        os.remove(img_path)

    # if len(result) == 1:
    #     pass
    # else:
    #     if result[0] is None and result[1] is None:
    #         print("Both are None")
    #         continue
    #     else:
    #         # replace this character image with 2 others
    #         # first_file = img_path.replace(".png", "_1.png")
    #         # second_file = img_path.replace(".png", "_2.png")
    #         first_file = f"characters/{filename}_1.png"
    #         second_file = f"characters/{filename}_2.png"
    #         cv2.imwrite(first_file, result[0])
    #         cv2.imwrite(second_file, result[1])
    #         # remove previos 
    #         replaced_files.append(filename)

print("Count of replaced files: ", len(replaced_files))
print("Replaced files: ", replaced_files)

    

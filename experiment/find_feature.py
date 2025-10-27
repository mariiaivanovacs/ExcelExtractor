#!/usr/bin/env python3
"""
compute_lr_columns.py

Generates synthetic images for characters '3' and '5', finds the character bounding
box, computes the first two left and first two right column intensities for each image,
prints results, writes annotated images, and saves a CSV summary.

Dependencies:
    pip install pillow numpy pandas

Run:
    python compute_lr_columns.py
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import pandas as pd
import csv

import warnings
warnings.filterwarnings('ignore')
from fight import downsample_then_upsample


# -------------------------
# Config
# -------------------------
OUTPUT_DIR = Path("generated_chars")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "left_right_intensities.csv"

CHARS = ["2", "8"]
IMAGES_PER_CHAR = 20
IMG_SIZE = (32, 32)            # (width, height)
BACKGROUND = 255               # white background (grayscale)
THRESHOLD = 250                # pixel < THRESHOLD considered ink (robust to anti-alias)
FONT_SIZES = (14, 16)          # range of random font sizes to use
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------
# Utilities
# -------------------------
def get_font(size):
    """Try common system fonts, fall back to PIL default if unavailable."""
    possible = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for p in possible:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_text_image(ch, img_size=IMG_SIZE, font_size=None):
    """Create a grayscale image of a single character with small random variations."""
    w, h = img_size
    if font_size is None:
        font_size = random.randint(*FONT_SIZES)
    font = get_font(font_size)

    canvas = Image.new("L", img_size, color=BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    # measure text size using textbbox (Pillow 10+ safe)
    bbox = draw.textbbox((0, 0), ch, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # random placement (keep inside canvas)
    max_tx = max(0, w - tw - 4)
    max_ty = max(0, h - th - 4)
    tx = random.randint(2, max(2, max_tx))
    ty = random.randint(2, max(2, max_ty))
    draw.text((tx, ty), ch, fill=0, font=font)  # black ink
    
    np_img = np.array(canvas).astype(np.float32) / 255.0
    
    import cv2

    scale = random.uniform(0.8, 0.83)
    small = cv2.resize(np_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    up = cv2.GaussianBlur(up, (3, 3), sigmaX=random.uniform(0.7, 1.0))
    np_img = up
    # np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)
    # soter for debugging
    random_int = random.randint(0, 1000000)
    cv2.imwrite(f"experiment/debug_out/{ch}_{random_int}.png", (np_img * 255).astype(np.uint8))
    arr = (np_img * 255).astype(np.uint8)

    # # Slight rotation and optional blur to simulate variation
    # angle = random.uniform(-10, 10)
    # canvas = canvas.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=BACKGROUND)
    # if random.random() < 0.5:
    #     canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.2)))

    # # Add light Gaussian noise
    # arr = np.array(canvas).astype(np.float32)
    # sigma = random.uniform(0, 5.0)
    # if sigma > 0.1:
    #     arr += np.random.normal(0, sigma, arr.shape)
    # arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_left_right_columns(img_arr, thresh=THRESHOLD):
    """
    Given a 2D numpy array (grayscale uint8), compute:
      - bounding box of pixels < thresh
      - left two columns indices (xmin, xmin+1) capped to bbox
      - right two columns indices (xmax, xmax-1) capped to bbox
      - column intensities computed as sum(255 - pixel_value) per column
    Returns:
      left_indices (list of 2 ints), left_sums (list of 2 ints),
      right_indices (list of 2 ints), right_sums (list of 2 ints),
      bbox (xmin, ymin, xmax, ymax)
    """
    h, w = img_arr.shape
    mask = img_arr < thresh
    cols_nonempty = np.where(mask.any(axis=0))[0]
    rows_nonempty = np.where(mask.any(axis=1))[0]

    if cols_nonempty.size == 0 or rows_nonempty.size == 0:
        # no ink found -> treat whole image as bbox (fallback)
        xmin, xmax = 0, w - 1
        ymin, ymax = 0, h - 1
    else:
        xmin, xmax = int(cols_nonempty[0]), int(cols_nonempty[-1])
        ymin, ymax = int(rows_nonempty[0]), int(rows_nonempty[-1])

    # left two columns: xmin and xmin+1 (use xmin twice if bbox is narrow)
    left_indices = [xmin, xmin + 1 if xmin + 1 <= xmax else xmin]
    # right two columns: xmax and xmax-1 (order: rightmost then the one left of it)
    right_indices = [xmax, xmax - 1 if xmax - 1 >= xmin else xmax]

    # ink measure: higher = more black. Use 255 - pixel_value for each pixel.
    ink = 255 - img_arr.astype(np.int32)
    left_sums = [int(ink[:, idx].sum()) for idx in left_indices]
    right_sums = [int(ink[:, idx].sum()) for idx in right_indices]

    return left_indices, left_sums, right_indices, right_sums, (xmin, ymin, xmax, ymax)


import numpy as np

def compute_top_bottom_rows(img_arr: np.ndarray, left_idx, right_idx, thresh=250):
    """
    Given a 2D numpy array (grayscale uint8), compute:
      - bounding box of pixels < thresh
      - top two row indices (ymin, ymin+1) capped to bbox
      - bottom two row indices (ymax, ymax-1) capped to bbox
      - row intensities computed as sum(255 - pixel_value) per row
    Returns:
      top_indices (list of 2 ints), top_sums (list of 2 ints),
      bottom_indices (list of 2 ints), bottom_sums (list of 2 ints),
      bbox (xmin, ymin, xmax, ymax)
    Notes:
      - If no ink is found, the bbox falls back to the whole image.
      - top_indices are ordered topmost then the next row down.
      - bottom_indices are ordered bottommost then the one above it.
    """
    img_arr = img_arr[:, left_idx[0]:right_idx[1]]
    h, w = img_arr.shape
    mask = img_arr < thresh
    cols_nonempty = np.where(mask.any(axis=0))[0]
    rows_nonempty = np.where(mask.any(axis=1))[0]

    if cols_nonempty.size == 0 or rows_nonempty.size == 0:
        # no ink found -> treat whole image as bbox (fallback)
        xmin, xmax = 0, w - 1
        ymin, ymax = 0, h - 1
    else:
        xmin, xmax = int(cols_nonempty[0]), int(cols_nonempty[-1])
        ymin, ymax = int(rows_nonempty[0]), int(rows_nonempty[-1])

    # top two rows: ymin and ymin+1 (use ymin twice if bbox is only one row tall)
    top_indices = [ymin, ymin + 1 if ymin + 1 <= ymax else ymin]
    # bottom two rows: ymax and ymax-1 (order: bottommost then the one above it)
    bottom_indices = [ymax, ymax - 1 if ymax - 1 >= ymin else ymax]
    
    img_arr = img_arr[top_indices[0]:bottom_indices[1], :]
        # ink measure: higher = more black. Use 255 - pixel_value for each pixel.
    ink = 255 - img_arr.astype(np.int32)
    row_sums = ink.sum(axis=1)  # sum across columns for each row
    col_sums = ink.sum(axis=0)  # sum across rows for each column

    
    h, w = img_arr.shape
    
    top_sums = [int(row_sums[idx]) for idx in [1,2]]
    bottom_sums = [int(row_sums[idx]) for idx in [h-2, h-1]]
    left_sums = [int(col_sums[idx]) for idx in [1,2]]
    right_sums = [int(col_sums[idx]) for idx in [w-2, w-1]]

    return top_sums, bottom_sums, left_sums, right_sums, top_indices, bottom_indices



def annotate_and_save_image(img, top_idx, top_sums, bot_idx, bot_sums,bbox,  out_path):
    """Create an RGB annotated copy that draws vertical lines for selected columns and writes values."""
    rgb = img.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    w, h = img.size

    # draw thin vertical lines at the columns
    for y in set(top_idx + bot_idx):
        draw.line([(0, y), (w, y)], width=1, fill=(255, 0, 0))

    # top text overlay (white background to ensure readability)
    # info = f"L idx:{left_idx} sums:{left_sums}  R idx:{right_idx} sums:{right_sums}  bbox:{bbox}"
    # draw.rectangle([(0, 0), (w, 14)], fill=(255, 255, 255))
    # draw.text((2, 0), info, fill=(0, 0, 0), font=get_font(10))

    rgb.save(out_path)
    
    
def visualize_features(three_left, three_right, five_left, five_right):
    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt
    import numpy as np

    three_left = np.array(three_left)
    three_right = np.array(three_right)
    five_left = np.array(five_left)
    five_right = np.array(five_right)

    plt.figure(figsize=(12, 6))

    # LEFT vs RIGHT for '3'
    plt.subplot(1, 2, 1)
    plt.boxplot([three_left, three_right], labels=["3 Left", "3 Right"])
    plt.title("Character 3 — Left vs Right Column Intensities")
    plt.ylabel("Intensity Sum")

    # LEFT vs RIGHT for '5'
    plt.subplot(1, 2, 2)
    plt.boxplot([five_left, five_right], labels=["5 Left", "5 Right"])
    plt.title("Character 5 — Left vs Right Column Intensities")
    plt.ylabel("Intensity Sum")

    plt.tight_layout()
    plt.show()

    # # RIGHT sums comparison
    # plt.subplot(1, 2, 2)
    # plt.boxplot([three_right[:,0], three_right[:,1], five_right[:,0], five_right[:,1]],
    #             labels=["3 Right1", "3 Right2", "5 Right1", "5 Right2"])
    # plt.title("Right Column Intensities (Characters 3 and 5)")
    # plt.ylabel("Intensity Sum")

    plt.tight_layout()
    plt.show()


import numpy as np

def split_character_into_8_areas(img_arr,
                                 left_idx, right_idx,
                                 top_idx, bottom_idx,
                                 measure='ink'):
    """
    Split the character bbox into 2 cols x 4 rows (8 areas) and compute average intensity
    for each area.

    Parameters
    ----------
    img_arr : 2D numpy array (grayscale, uint8 or numeric)
    left_idx, right_idx : int  (column indices, inclusive)
    top_idx, bottom_idx : int  (row indices, inclusive)
    measure : 'ink' (default) or 'pixel'
        'ink' -> uses (255 - pixel) so darker => larger value
        'pixel' -> uses raw pixel values (0..255)

    Returns
    -------
    averages : list of 8 floats
        Order: [L_top0, L_top1, L_top2, L_bottom, R_top0, R_top1, R_top2, R_bottom]
        (left column top->bottom then right column top->bottom)
    bboxes  : list of 8 tuples (x0, y0, x1, y1) inclusive coordinates for each area
    """
    # Validate input array
    if img_arr.ndim != 2:
        raise ValueError("img_arr must be a 2D grayscale array")

    h, w = img_arr.shape

    # Ensure integer indices and clamp to image bounds
    xmin = int(round(left_idx))
    xmax = int(round(right_idx))
    ymin = int(round(top_idx))
    ymax = int(round(bottom_idx))

    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w - 1))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h - 1))

    # If user gave indices reversed, swap to ensure xmin <= xmax, ymin <= ymax
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    # If bbox degenerate (no pixels) return zeros
    if width <= 0 or height <= 0:
        return [0.0] * 8, [(xmin, ymin, xmax, ymax)] * 8

    # Compute split sizes with remainder distributed to earlier parts
    # Columns: 2 parts
    qx, rx = divmod(width, 2)
    col_sizes = [qx + (1 if i < rx else 0) for i in range(2)]

    # Rows: 4 parts
    qy, ry = divmod(height, 4)
    row_sizes = [qy + (1 if i < ry else 0) for i in range(4)]

    # Build start indices (inclusive) for columns and rows
    x_starts = [xmin]
    for s in col_sizes[:-1]:
        x_starts.append(x_starts[-1] + s)
    x_ends = [x_starts[i] + col_sizes[i] - 1 for i in range(2)]

    y_starts = [ymin]
    for s in row_sizes[:-1]:
        y_starts.append(y_starts[-1] + s)
    y_ends = [y_starts[i] + row_sizes[i] - 1 for i in range(4)]

    # Choose value array
    if measure == 'ink':
        vals = (255 - img_arr.astype(np.int32))
    elif measure == 'pixel':
        vals = img_arr.astype(np.float64)
    else:
        raise ValueError("measure must be 'ink' or 'pixel'")

    averages = []
    bboxes = []

    # Order: left column (col 0) top->bottom (rows 0..3), then right column (col 1) top->bottom
    for col_idx in range(2):
        x0 = x_starts[col_idx]
        x1 = x_ends[col_idx]
        start = 0
        previous = None
        for row_idx in range(4):
            y0 = y_starts[row_idx]
            y1 = y_ends[row_idx]

            # defensive check: if any dimension 0-sized, return 0.0
            if x1 < x0 or y1 < y0:
                avg = 0.0
            else:
                patch = vals[y0:y1+1, x0:x1+1]
                if patch.size == 0:
                    avg = 0.0
                else:
                    avg = float(patch.mean())
            if not previous:
                averages.append(start)
                previous = round(avg, 3)
            else:
                if previous < round(avg, 3):
                    start += 1
                else:
                    start -= 1
                averages.append(start)
                previous = round(avg, 3)


            bboxes.append((x0, y0, x1, y1))

    return averages, bboxes


arr_1_first, arr_2_first, arr_3_first, arr_4_first, arr_5_first, arr_6_first, arr_7_first, arr_8_first = [], [], [], [], [], [], [], []
arr_1_second, arr_2_second, arr_3_second, arr_4_second, arr_5_second, arr_6_second, arr_7_second, arr_8_second = [], [], [], [], [], [], [], []


import cv2
# -------------------------
# Main generation + computation loop
# -------------------------
def main():
    rows = []
    print("Generating images and computing left/right column intensities...")
    for ch in CHARS:

        for i in range(IMAGES_PER_CHAR):
            img = generate_text_image(ch)
            fname = f"{ch}_{i:02d}.png"
            fpath = OUTPUT_DIR / fname
            img.save(fpath)

            arr = np.array(img)  # 2D array
            left_idx, left_sums, right_idx, right_sums, bbox = compute_left_right_columns(arr)
            
            
            top_sums, bottom_sums, left_sums, right_sums, top_idx, bot_idx = compute_top_bottom_rows(arr, left_idx, right_idx, thresh=250,)
            # print("top rows:", top_idx, "sums:", top_sums)
            # print("bottom rows:", bot_idx, "sums:", bot_sums)
            
            
            left, right, top, bottom = left_idx[0], right_idx[1], top_idx[0], bot_idx[1]
            avgs, boxes = split_character_into_8_areas(arr, left, right, top, bottom)
            
            if ch == "2":
                arr_1_first.append(avgs[0])
                arr_2_first.append(avgs[1])
                arr_3_first.append(avgs[2])
                arr_4_first.append(avgs[3])
                arr_5_first.append(avgs[4])
                arr_6_first.append(avgs[5])
                arr_7_first.append(avgs[6])
                arr_8_first.append(avgs[7])               
            if ch == "8":
                arr_1_second.append(avgs[0])
                arr_2_second.append(avgs[1])
                arr_3_second.append(avgs[2])
                arr_4_second.append(avgs[3])
                arr_5_second.append(avgs[4])
                arr_6_second.append(avgs[5])
                arr_7_second.append(avgs[6])
                arr_8_second.append(avgs[7])
            gray = arr
            # var_1 = float(np.linalg.norm(arr))
            height, width = arr.shape
            area = height * width
            # # 4. Canny edge & density
            # canny = cv2.Canny(arr, 100, 200)
            # var_2 = float(np.sum(canny > 0) / area)
            # var_1 = np.sum(gray, axis=0) / float(height)
            # var_2= np.sum(gray, axis=1) / float(width)
            
# 14. Projection correlation - correlation between row and column projections
    # Resize projections to same length for correlation calculation
                # 5/6. Column / Row intensity projections
            col_intensity = np.sum(gray, axis=0) / float(height)
            row_intensity = np.sum(gray, axis=1) / float(width)
            min_len = min(len(row_intensity), len(col_intensity))
            row_resized = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(row_intensity)), row_intensity)
            col_resized = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(col_intensity)), col_intensity)

            # 18. Projection entropy ratio - Shannon entropy of row vs column intensities
            def calculate_entropy(values):
                """Calculate Shannon entropy of a distribution."""
                # Normalize to create probability distribution
                values = np.array(values) + 1e-12  # avoid log(0)
                probs = values / np.sum(values)
                # Calculate entropy
                entropy = -np.sum(probs * np.log2(probs + 1e-12))
                return entropy

            row_entropy = float(calculate_entropy(row_intensity))
            col_entropy = float(calculate_entropy(col_intensity))
            projection_entropy_ratio = float(row_entropy / (col_entropy + 1e-12))
            # var_1 = row_entropy
            # var_2 = col_entropy


            # if ch == "1":
            #     # first_moment = float(np.mean(arr))
            #     # second_moment = float(np.var(arr))
            #      # 3. var_1 (Frobenius norm)

            #     three_left.append(var_1)
            #     three_right.append(var_2)
            # if chan == "0":
            #     five_left.append(var_1)
            #     five_right.append(var_2)
                

            # Save annotated image for visual inspection
            anno_name = f"{ch}_{i:02d}_annot.png"
            annotate_and_save_image(img, top_idx, top_sums, bot_idx, bottom_sums, bbox, OUTPUT_DIR / anno_name)

            # Append to CSV rows
            rows.append({
                "filename": fname,
                "char": ch,
                "bbox_xmin": bbox[0], "bbox_ymin": bbox[1], "bbox_xmax": bbox[2], "bbox_ymax": bbox[3],
                "left_col1_idx": left_idx[0], "left_col1_sum": left_sums[0],
                "left_col2_idx": left_idx[1], "left_col2_sum": left_sums[1],
                "right_col1_idx": right_idx[0], "right_col1_sum": right_sums[0],
                "right_col2_idx": right_idx[1], "right_col2_sum": right_sums[1],
                "annotated_file": anno_name
            })
            
    
            
    # visualize_features(three_left, three_right, five_left, five_right)
       
    first_arrays = [arr_1_first, arr_2_first, arr_3_first, arr_4_first, arr_5_first, arr_6_first, arr_7_first, arr_8_first] 
    second_arrays = [arr_1_second, arr_2_second, arr_3_second, arr_4_second, arr_5_second, arr_6_second, arr_7_second, arr_8_second]      
    # visualize_features(three_left, three_right, five_left, five_right)
    print(f"FIRST CHARACTER: {CHARS[0]}")
    count = 0
    for i_array in first_arrays:
        print(f"AVErage of area {count}: {np.mean(i_array)}")
        count += 1
        
    print(f"SECOND CHARACTER: {CHARS[1]}")
    count = 0
    for i_array in second_arrays:
        print(f"AVErage of area {count}: {np.mean(i_array)}")
        count += 1
        
    
        
    


    
    # Write CSV
    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nDone. Saved {len(rows)} rows to {CSV_PATH}")
    print(f"Annotated images and originals are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

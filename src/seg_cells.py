# segment.py
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
# improt random
import random
import matplotlib.pyplot as plt

from try_blobs import extract_blobs

# i need to import from the directory o nthe same level as parent
import sys
import os

# Add the parent directory to sys.path so 'utils' is importable
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
    bright_thresh = 0.87
    color_thresh = 0.1

    # Mask for light gray pixels
    mask = (brightness > bright_thresh) & (color_diff < color_thresh)

    # Replace with white
    img[mask] = [1.0, 1.0, 1.0]

    # Convert back to uint8
    result = (img * 255).astype(np.uint8)
    
    
    cv2.imwrite(output_path, result)
    return result
# from math import random

# -------------------------
# Helper utils / defaults
# -------------------------
DEFAULTS = {
    "area_min": 40,          # minimal connected-component area to keep
    "min_h": 8,              # minimal height in px to keep
    "max_ratio": 4.0,        # if w/h > this, try splitting
    "wide_split_ratio": 2.5, # if w/h > this -> split by projection
    "gap_rel": 0.7,          # gap threshold relative to avg char width
    "morph_kernel": (2, 2),
    "use_adaptive_thresh": True,
    "adaptive_block": 15,
    "adaptive_C": 8,
}

# -------------------------
# Core segmentation
# -------------------------
def segment_cell(img_cell: np.ndarray, params: Dict = None) -> List[Dict]:
    """
    Segment characters inside a cell image.

    Args:
        img_cell: np.ndarray BGR or grayscale image of the whole cell.
        params: optional dict of parameters (overrides DEFAULTS).

    Returns:
        List of dicts sorted by left-to-right with keys:
            - 'bbox': (x, y, w, h)
            - 'crop': grayscale crop (numpy array)
            - 'center_x': center x coordinate (int)
            - 'mask': binary mask of the crop (optional)
    """
    if params is None:
        params = DEFAULTS
    else:
        p = DEFAULTS.copy()
        p.update(params)
        params = p

    # ensure grayscale
    if img_cell.ndim == 3:
        gray = cv2.cvtColor(img_cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cell.copy()

    # 1) Denoise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2) Binarize: adaptive if requested, else Otsu
    if params["use_adaptive_thresh"]:
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       params["adaptive_block"],
                                       params["adaptive_C"])
    else:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) Morphology (clean small holes / specks)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params["morph_kernel"])
    # first open to remove small dots, then close to fill small holes
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4) Find contours / connected components
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    h_img, w_img = gray.shape

    for c in contours:
        print("CONTOUR IS FOUND")
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # filter small/noise
        # if area < params["area_min"] or h < params["min_h"]:
        #     print("CONTOUR IS TOO SMALL")
        #     continue
        boxes.append((x, y, w, h))
    print("AFTER LEN OF BOXES:")
    print(len(boxes))

    # 5) Expand boxes slightly (optional) to include anti-aliased edges
    expanded = []
    pad = 1
    for x, y, w, h in boxes:
        xa = max(0, x - pad)
        ya = max(0, y - pad)
        xb = min(w_img, x + w + pad)
        yb = min(h_img, y + h + pad)
        expanded.append((xa, ya, xb - xa, yb - ya))

    # 6) For each box: if very wide -> try splitting by projection / watershed
    final_boxes = []
    for (x, y, w, h) in expanded:
        if w / max(1, h) > params["wide_split_ratio"]:
            splits = split_wide_box_by_projection(morph[y:y+h, x:x+w], x_offset=x,
                                                 gap_rel=params["gap_rel"])
            if splits:
                final_boxes.extend(splits)
                continue
            # fallback to watershed split
            ws = split_wide_box_by_watershed(morph[y:y+h, x:x+w], x_offset=x)
            if ws:
                final_boxes.extend(ws)
                continue
        final_boxes.append((x, y, w, h))

    # 7) Post-filter: remove nested / duplicates (keep largest)
    final_boxes = non_max_suppression_boxes(final_boxes, iou_thresh=0.15)

    # 8) Create output entries and sort left-to-right
    entries = []
    for (x, y, w, h) in final_boxes:
        crop_gray = gray[y:y+h, x:x+w]
        crop_mask = morph[y:y+h, x:x+w]
        center_x = x + w // 2
        entries.append({
            "bbox": (x, y, w, h),
            "crop": crop_gray,
            "mask": crop_mask,
            "center_x": center_x
        })

    entries = sorted(entries, key=lambda e: e["center_x"])
    return entries

# -------------------------
# Splitting helpers
# -------------------------
def split_wide_box_by_projection(bin_crop: np.ndarray, x_offset: int = 0, gap_rel: float = 0.7):
    """
    Split an input binary crop (foreground = 255) vertically using column projection.
    Returns list of bboxes (x,y,w,h) with global coordinates (x offset added).
    """
    h, w = bin_crop.shape
    # sum of foreground per column (foreground is 255 for inverted thresh)
    col_sum = (bin_crop > 0).astype(np.int32).sum(axis=0)
    # smooth column sum to avoid tiny spikes
    kernel = np.ones(3, dtype=np.int32)
    col_sum_s = np.convolve(col_sum, kernel, mode='same')

    # estimate average char width by detecting connected groups in projection
    nonzero = np.where(col_sum_s > 0)[0]
    if nonzero.size == 0:
        return []

    # find gaps where col_sum == 0 or very small
    zero_mask = col_sum_s <= (col_sum_s.max() * 0.05)  # columns with near-zero ink
    # find continuous zero intervals
    gaps = []
    i = 0
    while i < w:
        if zero_mask[i]:
            j = i
            while j < w and zero_mask[j]:
                j += 1
            gaps.append((i, j))
            i = j
        else:
            i += 1

    # determine segment boundaries using gaps that are wide enough
    # estimate average width by looking at widths between gaps
    # build non-gap segments
    segments = []
    prev = 0
    for (a, b) in gaps:
        if a - prev > 2:
            segments.append((prev, a))
        prev = b
    if prev < w:
        segments.append((prev, w))

    # if we get multiple segments, convert to boxes
    boxes = []
    for (sx, ex) in segments:
        seg_w = ex - sx
        if seg_w <= 2:
            continue
        boxes.append((x_offset + sx, 0, seg_w, h))

    # if segmentation produced only one segment — return empty (no split)
    if len(boxes) <= 1:
        return []
    return boxes

def split_wide_box_by_watershed(bin_crop: np.ndarray, x_offset: int = 0):
    """
    Attempt to split connected characters inside a wide binary crop using watershed.
    Returns list of bboxes in global coords.
    """
    # bin_crop: binary (255 foreground)
    # convert to uint8
    img = bin_crop.copy()
    if img.max() <= 1:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    # distance transform
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    # threshold to obtain sure foreground
    _, fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    # sure background
    kernel = np.ones((3,3), np.uint8)
    bg = cv2.dilate(img, kernel, iterations=3)
    # markers
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[bg == 0] = 0  # background should be 0 for watershed

    # prepare color image for watershed
    h, w = img.shape
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        markers = cv2.watershed(color, markers)
    except Exception:
        return []

    # regions >1 correspond to unique markers
    boxes = []
    for m in np.unique(markers):
        if m <= 1:
            continue
        mask = (markers == m).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        x, y, w_box, h_box = cv2.boundingRect(cnts[0])
        boxes.append((x_offset + x, y, w_box, h_box))

    # keep boxes sorted left-to-right
    boxes = sorted(boxes, key=lambda b: b[0])
    # if nothing sensible, return empty
    if len(boxes) <= 1:
        return []
    return boxes

# -------------------------
# NMS for small boxes
# -------------------------
def non_max_suppression_boxes(boxes: List[Tuple[int,int,int,int]], iou_thresh: float = 0.15):
    """
    Removes nested or heavily overlapping boxes. Returns filtered boxes.
    Boxes are (x,y,w,h).
    """
    if not boxes:
        return []
    arr = np.array(boxes).astype(np.float32)
    x1 = arr[:,0]
    y1 = arr[:,1]
    x2 = arr[:,0] + arr[:,2]
    y2 = arr[:,1] + arr[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = x1.argsort()  # left to right order

    keep = []
    used = np.zeros(len(boxes), dtype=np.bool_)
    for i in order:
        if used[i]:
            continue
        keep.append(tuple(arr[i].astype(int)))
        for j in range(len(boxes)):
            if used[j] or j == i:
                continue
            # compute IoU
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w_int = max(0.0, xx2 - xx1 + 1)
            h_int = max(0.0, yy2 - yy1 + 1)
            inter = w_int * h_int
            union = areas[i] + areas[j] - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou > iou_thresh:
                # mark smaller one as used
                if areas[i] >= areas[j]:
                    used[j] = True
                else:
                    used[i] = True
    return keep

# -------------------------
# Visualization helper
# -------------------------
def draw_boxes(img: np.ndarray, entries: List[Dict], color=(0,255,0), thickness=1):
    """
    Draw bounding boxes and index on a color image copy.
    Returns annotated image (BGR).
    """
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    for i, e in enumerate(entries):
        x,y,w,h = e["bbox"]
        cv2.rectangle(out, (x,y), (x+w, y+h), color, thickness)
        cx = e["center_x"]
        cv2.putText(out, str(i), (x, max(y-3,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def upscale_image(img, factor):
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor))

# -------------------------
# Preprocess for classifier
# -------------------------
def preprocess_for_model(crop: np.ndarray, target_size: int = 28, pad: int = 2):
    """
    Resize crop to target_size x target_size keeping aspect ratio, center by mass.
    Returns normalized float32 array in range [0,1] with foreground=1.
    """
    # crop is grayscale
    h, w = crop.shape
    # threshold to get binary for bbox of ink
    _, b = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(b < 255)
    if ys.size > 0:
        minx, maxx = xs.min(), xs.max()
        miny, maxy = ys.min(), ys.max()
        roi = crop[miny:maxy+1, minx:maxx+1]
    else:
        roi = crop

    # scale preserving aspect ratio
    h2, w2 = roi.shape
    scale = (target_size - pad*2) / max(h2, w2)
    new_w = max(1, int(w2 * scale))
    new_h = max(1, int(h2 * scale))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # place centered on white canvas
    canvas = np.ones((target_size, target_size), dtype=np.uint8) * 255
    x0 = (target_size - new_w) // 2
    y0 = (target_size - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized

    arr = canvas.astype(np.float32)
    arr = 1.0 - (arr / 255.0)  # foreground high
    return arr

# -------------------------
# New character separation logic
# -------------------------
def remove_cell_boundaries(img: np.ndarray, boundary_thickness: int = 10) -> np.ndarray:
    """
    Remove black boundaries from cell image by setting border pixels to white.

    Args:
        img: Grayscale cell image
        boundary_thickness: Number of pixels to remove from each border

    Returns:
        Image with boundaries removed
    """
    result = img.copy()
    h, w = result.shape

    # Set border pixels to white (255)
    result[:boundary_thickness, :] = 255  # Top
    result[-boundary_thickness+3:, :] = 255  # Bottom
    result[:, :boundary_thickness] = 255  # Left
    result[:, -boundary_thickness:] = 255  # Right
    
    # save result for debug
    # random_int = random.randint(0, 1000000)
    # cv2.imwrite(f"results/cell_no_boundaries_{random_int}.png", result)

    return result



def find_top_character_profile(img_float: np.ndarray, top_fraction: float = 0.5, threshold: float = 0.5):
    """
    Scan the top `top_fraction` of the image to detect, for each column, 
    the first row where intensity > threshold.

    Args:
        img_float: 2D float array, range [0,1].
        top_fraction: fraction of rows from the top to scan (e.g., 0.3 = top 30%).
        threshold: intensity threshold that defines "character ink".

    Returns:
        fin_character: 1D float array of length W, normalized in [0,1]
                       (1 = top, 0 = bottom, 0 if no ink found).
    """
    
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 2, 1)
    # plt.imshow(img_float, cmap='gray')
    # cv2.imwrite("results/img_float.png", img_float)
    H, W = img_float.shape
    max_depth = int(H * top_fraction)
    max_depth_top = int(H * top_fraction)
    print(f"max_depth: {max_depth}")
    
    
    fin_character = np.array([-1]*W)# each element = row index where ink starts (0 if none)

    for row_idx in range(max_depth):
        row = img_float[row_idx, :]
        for i in range(len(row)):
            if row[i] > threshold and fin_character[i] == -1:
                fin_character[i] =row_idx
                
        
        # ink_mask = row > threshold  # columns where ink appears at this row
        # for i in range(len(ink_mask)):
        #     if ink_mask[i] and fin_character[i] == -1:
                
        #         fin_character[i] = row_idx + 1
        # print(f"Current fin_character: {fin_character}")
        # We only set the position if it hasn’t been set before (==0)
        # unset = fin_character == 0
        # fin_character[unset & ink_mask] = row_idx + 1  # +1 avoids confusion with 0 (meaning “not found yet”)
        # fin_character = np.zeros(W, dtype=float)  # stores the *row index* where ink starts
        # print(f"Fin character: {fin_character}")
        # Normalize: convert row index → vertical scale [0,1]
        # (top=1.0, bottom=0.0)
        # Note: if no ink found in a column, fin_character[col]=0 → stays 0
        # fin_character = np.clip((max_depth - fin_character) / max_depth, 0, 1)
    fin_character_top = np.clip((H - fin_character) / H, 0.5, 1)

    # print(f"Fin character top: {fin_character_top}")
    return fin_character_top




def calculate_column_intensity(img: np.ndarray, number=1) -> np.ndarray:
    """
    Calculate average intensity for each column of pixels.
    For text detection, we want high values where there's text content.

    Args:
        img: Grayscale image

    Returns:
        Array of average intensities for each column (normalized 0-1)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to float and normalize to 0-1 range
    img_float = img.astype(np.float32) / 255.0
    
    # convert to 3 np arrays one for top 30% fo rows and one for bottom 30% of rows and one for middle 40% of rows
    h, w = img_float.shape
    top_30 = img_float[:int(h*0.3), :]
    bottom_30=img_float[-int(h*0.3):, :]
    middle_40 = img_float[int(h*0.3):int(h*0.7), :]

    # bottom_30=find_top_character_profile(img_float)
    
    
    max_depth = int(h* 0.3)
    fin_character = img_float[0] > 0.5
    for i in range(max_depth):
        row = img_float[i]
        is_finished = row > 0.5
        
    
    
    # Calculate average intensity per column (inverted: dark = high)
    inverted = 1.0 - img_float
    column_intensities = np.mean(inverted, axis=0)
    column_intensities = column_intensities 

    # Threshold for visualization
    threshold = 0.1
    content_cols_normal = np.sum(column_intensities < (1.0 - threshold)) # Dark content
    

    # Compute mean intensity across all columns
    mean_intensity = np.mean(column_intensities) / 2
    
    

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(column_intensities, label='Column Intensities', color='blue')
    plt.title('Column Intensities (Normal)')
    plt.xlabel('Column')
    plt.ylabel('Average Intensity')

    # Add threshold line
    plt.axhline(y=1.0 - threshold, color='r', linestyle='--', label=f'Threshold {1.0 - threshold}')

    # Add mean line (dark yellow)
    plt.axhline(y=mean_intensity, color='#b58900', linestyle='-', linewidth=2, label=f'Mean = {mean_intensity:.3f}')

    plt.legend()
    plt.tight_layout()

    output_path = f"results/inspec_{number}.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

    return column_intensities

def calculate_column_intensity_top_bottom(img: np.ndarray,
                                          top_frac: float = 0.30,
                                          bottom_frac: float = 0.30,
                                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate column intensities for the top and bottom fractions of the image,
    plot them (with mean lines), and return the two column-intensity arrays.

    Args:
        img: Grayscale image (HxW) with values in [0,255] or [0,1].
        top_frac: fraction of image height to take from the top (default 0.30).
        bottom_frac: fraction of image height to take from the bottom (default 0.30).
        save_path: where to save the visualization.

    Returns:
        (column_intensity_top, column_intensity_bottom) — each is a 1D np.ndarray of length W,
        normalized to [0,1] where larger values indicate darker ink (inverted intensity).
    """ 
    random_int=random.randint(0, 1000000)  
    save_path: str = f"results/inspect_top_bottom_{random_int}.png"
    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to float and normalize to 0-1 range
    img_float = img.astype(np.float32) / 255.0
    cv2.imwrite(f"results/img_float_{random_int}.png", img)
    
    print(f"Image float shape is here: {img_float.shape}")
    max_depth = 0.3 
    
    # convert to 3 np arrays one for top 30% fo rows and one for bottom 30% of rows and one for middle 40% of rows
    h, w = img_float.shape
    top_30 = img_float[:int(h*0.3), :]
    bottom_30=img_float[-int(h*0.3):, :]
    middle_40 = img_float[int(h*0.3):int(h*0.7), :]

    # bottom_30=find_top_character_profile(img_float)
    middle_40 = img_float[int(h*0.3):int(h*0.7), :]
    
    
    

    # Calculate average intensity per column (inverted: dark = high)
    inverted = 1.0 - img_float
    # column_intensities = np.mean(inverted, axis=0)
    column_intensities = np.mean(inverted, axis=0)   # shape (w,)

    # top/bottom crops
    top_region = img_float[: int(h * 0.5), :]           # shape (h_top, w)
    top_upper_region = img_float[: int(h* 0.3), :]     # shape (h_top_upper, w)
    avg_top_region = np.mean(top_region, axis=0)       # shape (w,)
    avg_top_upper_region = np.mean(top_upper_region, axis=0)  # shape (w,)
    top_region_avg = (avg_top_region + avg_top_upper_region) / 2  # shape (w,)

    # bottom uses inverted (so 'ink' is positive)
    bottom_region = inverted[-int(h * 0.5):, :]         # shape (h_bottom, w)
    bottom_lower_region = inverted[-int(h * 0.3):, :]   # shape (h_bottom_lower, w)
    avg_bottom_region = np.mean(bottom_region, axis=0)           # shape (w,)
    avg_bottom_lower_region = np.mean(bottom_lower_region, axis=0)  # shape (w,)
    bottom_region_avg = (avg_bottom_region + avg_bottom_lower_region) / 2  # shape (w,)

    # --- IMPORTANT FIX: do NOT take mean again across axis=0 if you want a per-column signal ---
    column_intensity_top = top_region_avg        # shape (w,)
    column_intensity_bottom = bottom_region_avg  # shape (w,)

    # scalar means (optional diagnostics)
    mean_top = float(np.mean(column_intensity_top))
    mean_bottom = float(np.mean(column_intensity_bottom))
    mean_full = float(np.mean(column_intensities))


    # --- Plotting ---
    plt.figure(figsize=(14, 8))

    # show original image on top-left
    plt.subplot(2, 2, 1)
    plt.imshow(img_float, cmap='gray', aspect='auto')
    plt.title('Original Image (normalized)')
    plt.axis('off')

    # show the top-region as an image for context
    plt.subplot(2, 2, 2)
    plt.imshow(top_region, cmap='gray', aspect='auto')
    plt.title(f'Top region')
    plt.axis('off')

    # main column intensity plot (bottom-left area of the grid)
    plt.subplot(2, 1, 2)
    x = np.arange(w)
    plt.plot(x, column_intensity_top, label='Top 30% Column Intensity', linewidth=1.2)
    plt.plot(x, column_intensity_bottom, label='Bottom 30% Column Intensity', linewidth=1.2)

    # mean lines (dark yellow for overall mean, dashed)
    plt.axhline(y=mean_full, color='#b58900', linestyle='-', linewidth=2,
                label=f'Overall Mean = {mean_full:.3f}')
    # also show top and bottom means as thinner dashed lines for clarity
    plt.axhline(y=mean_top, color='C0', linestyle='--', linewidth=1, label=f'Top Mean = {mean_top:.3f}')
    plt.axhline(y=mean_bottom, color='C1', linestyle='--', linewidth=1, label=f'Bottom Mean = {mean_bottom:.3f}')

    plt.title('Column Intensities (Top vs Bottom)')
    plt.xlabel('Column index')
    plt.ylabel('Average inverted intensity (dark = high)')
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()

    # save and report
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to: {save_path}")

    return column_intensity_top, column_intensity_bottom



# def find_character_boundaries(column_intensities: np.ndarray, threshold: float = 0.2,
#                              min_char_width: int = 5, min_gap_width: int = 2) -> List[Tuple[int, int]]:
#     """
#     Find character boundaries based on column intensity patterns.

#     Args:
#         column_intensities: Array of column intensities
#         threshold: Threshold below which a column is considered a separator
#         min_char_width: Minimum width for a valid character
#         min_gap_width: Minimum width for a valid gap between characters

#     Returns:
#         List of (start, end) tuples for each character region
#     """
#     # Find columns that are above threshold (contain character content)
#     above_threshold = column_intensities > threshold

#     # Find transitions from low to high intensity (character starts)
#     # and from high to low intensity (character ends)
#     boundaries = []
#     in_character = False
#     start_col = 0

#     for i, is_content in enumerate(above_threshold):
#         if not in_character and is_content:
#             # Start of character
#             start_col = i
#             in_character = True
#         elif in_character and not is_content:
#             # End of character
#             char_width = i - start_col
#             if char_width >= min_char_width:
#                 boundaries.append((start_col, i))
#             in_character = False

#     # Handle case where character extends to end of image
#     if in_character:
#         char_width = len(column_intensities) - start_col
#         if char_width >= min_char_width:
#             boundaries.append((start_col, len(column_intensities)))

#     # Merge boundaries that are too close together (likely same character)
#     if len(boundaries) > 1:
#         merged_boundaries = []
#         current_start, current_end = boundaries[0]

#         for start, end in boundaries[1:]:
#             gap_width = start - current_end
#             if gap_width < min_gap_width:
#                 # Merge with current character
#                 current_end = end
#             else:
#                 # Save current character and start new one
#                 merged_boundaries.append((current_start, current_end))
#                 current_start, current_end = start, end

#         # Add the last character
#         merged_boundaries.append((current_start, current_end))
#         boundaries = merged_boundaries

#     return boundaries


def find_character_boundaries_by_columns(
     img: np.ndarray,
        column_intensities: np.ndarray,
        threshold: float = 0.1,
        min_char_width: int = 2,
        min_gap_width: int = 1,
        return_states: bool = False,
        boundary: int = 2.5
    ) -> Tuple[List[Tuple[int, int]], int, Optional[np.ndarray]]:
    """
    Find character boundaries by scanning each column and grouping consecutive
    'in_character' columns into runs.

    Args:
        column_intensities: 1D array of column intensities (preferably in [0,1]).
        threshold: column is considered 'in_character' when its intensity > threshold.
        min_char_width: minimum width (in columns) for a run to be considered a character.
        min_gap_width: if the gap between two valid runs is < min_gap_width they will be merged.
        return_states: if True, also return the boolean per-column `in_character` array.

    Returns:
        (boundaries, count, states)
        - boundaries: list of (start, end) tuples, end is exclusive (Python-slice friendly).
        - count: number of character boundaries after merging.
        - states: optional 1D boolean array of length = n_columns (True = in_character).
    """
    
    # baseline = 0.01
    # col = np.asarray(column_intensities).astype(float).ravel()
    # n = col.size
    # if n == 0:
    #     return [], 0, (np.array([], dtype=bool) if return_states else None)

    # # Normalize to [0,1] if values appear to be in a larger scale (e.g., 0..255)
    # if col.max() > 1.0:
    #     col = col / float(col.max())

    # # Per-column boolean: True if column likely contains ink/character
    # # intensity_df = pd.DataFrame(column_intensities, columns=['intensity'])
    # # states = intensity_df['intensity'] > threshold
    # # mean_intensity = np.mean(column_intensities)
   
    
    # # mean_intensity = np.mean(column_intensities)
    # mean_intensity = np.mean(column_intensities) / 2
    # print(f"Mean value is: {mean_intensity}")

    # states = col > threshold
    # print(f"COL: {col}")
    # print(f"States: {states}")
    
        # --- validate / normalize inputs ---
        
    h, w = img.shape
    print(f"Image shape: {img.shape}")
    col = np.asarray(column_intensities).astype(float).ravel()
    n = col.size
    if n == 0:
        return [], 0, (np.array([], dtype=bool) if return_states else None)

    # Normalize to [0,1] if values appear to be in a larger scale (e.g., 0..255)
    if col.max() > 1.0:
        col = col / float(col.max())

    baseline = 0.01
    gray = img
    if gray.max() > 1.0:
        gray = gray / 255.0
    gray = np.clip(gray, 0.0, 1.0)
    inverted = 1 - gray

    # calculate criteria to identify black pixel
    h_1 = int((h /2) -5)
    h_2 =int((h /2) +5)
    w_1 = int((w /2) -5)
    w_2 = int((w /2) +5)
    central_matrix = gray[h_1:h_2, w_1:w_2]  # (height: 11, width: 11)
    flattened = central_matrix.flatten()

    # Sort ascending (darkest first)
    sorted_vals = np.sort(flattened)

    # Take top 10 darkest pixels (smallest intensity)
    top10_dark = sorted_vals[:10]

    # If you prefer to express "blackness", invert values to [0..1], so darker = higher blackness
    blackness = 1 - (top10_dark / 255.0)

    avg_top10_blackness = np.mean(blackness) / boundary # we use only half
    print(f"MAXES BLACKS: {avg_top10_blackness}")
                
         # calculate average col intensity
       
    cc=col[95:106]
    print(cc)
    half_avg_top10 = sum(cc)/len(cc)
    
    print(f"Half average 10 blackness: {half_avg_top10}")
    print(f"Average blackness: {avg_top10_blackness}")
                
    # blackness: 1.0 for black, 0.0 for white
    # --- determine per-column states according to your rule ---
    states = np.zeros(w, dtype=bool)

    for c in range(w):
        col_val = float(col[c])

        if col_val < half_avg_top10:
            # print()
        
            cur_arr = inverted[h_1:h_2, c]
            if np.any(cur_arr >= avg_top10_blackness):
                print(np.max(cur_arr))
                
                states[c] = True
                print("FOUND CONTENT")
            else:
                states[c] = False
            print("FOUND SPACE")
        else:
            # column intensity >= half_avg_top10 => likely ink present
            states[c] = True
    boundaries: List[Tuple[int, int]] = []
    in_character = False
    run_start = 0
    
    print(states)

    # print(f"Half average 10 blackness: {half_avg_top10}")
    

    small_punc_character = None
    # 1) Scan columns and group consecutive True states into runs
    for i, is_content in enumerate(states):
        if small_punc_character:
            if i in small_punc_character:
                pass
                # boundaries.append((small_punc_character[0], small_punc_character[1]))
            else:
                print(f"End of small punc character at column {i}")
                boundaries.append((small_punc_character[0], small_punc_character[9]))
                print(f"Boundaries: {boundaries}")
                small_punc_character = None
                in_character = False
    
        else:    
        
            if is_content:
                if i - run_start >= 12:
                    print("A bit big gap")
                    
                #     # look for closest light columns
                #     cols = col[i-2:i+2]
                #     lightest_col_index = np.argmin(cols) + (i-2)
                #     # finish character
                #     run_end = lightest_col_index 
                #     run_width = run_end - run_start
                #     if run_width >= min_char_width:
                #         boundaries.append((run_start, run_end))
                #     # reset
                #     in_character = False
                if not in_character:
                    # start a new run
                    run_start = i
                    in_character = True
                
                # else: continuing current run
            else:
                if in_character:
                    # end of run at column i (exclusive)
                    run_end = i
                    run_width = run_end - run_start
                    # print(f"run_width: {run_width}")
                    # print(f"min_char_width: {min_char_width}")
                    if run_width >= min_char_width:
                        boundaries.append((run_start, run_end))
                    # reset
                    in_character = False
                if (col[i] + col[i-1] + col[i-2])/3 <= baseline:
                    #    look for next 4 cols, if the value is increasing but not by much, consider it a character
                    next_growth = col[i+1:i+10].mean() - col[i]
                    if next_growth > 0.02 and next_growth < 0.05:
                        max_value = col[i+1:i+10].max()
                        if max_value <0.05:
                            print(f"Found content at column {i} with value {next_growth}")

                            print(f"FOUND DOT")
                            small_punc_character = range(i, i+10)
                            print(f"small_punc_character: {len(small_punc_character)}")
                            
                            
                    

    # if last run reaches the last column
    if in_character:
        run_end = n
        run_width = run_end - run_start
        if run_width >= min_char_width:
            boundaries.append((run_start, run_end))

    # 2) Merge runs separated by tiny gaps (< min_gap_width)
    if len(boundaries) > 1 and min_gap_width > 0:
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = boundaries[0]
        for s, e in boundaries[1:]:
            gap = s - cur_e
            if gap < min_gap_width:
                # merge this run into current
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        boundaries = merged

    count = len(boundaries)
    return boundaries, count, (states if return_states else None)

def check_vertical_line_clear(img: np.ndarray, col: int) -> bool:
    """
    Check if a vertical line at given column has no black pixels.

    Args:
        img: Grayscale image
        col: Column index to check

    Returns:
        True if line is clear (no black pixels), False otherwise
    """
    if col < 0 or col >= img.shape[1]:
        return False

    # Check if all pixels in the column are light (above threshold)
    column_pixels = img[:, col]
    # Consider pixels below 128 as "black" content
    return np.all(column_pixels > 128)

def crop_character_vertically(char_img: np.ndarray) -> np.ndarray:
    """
    Crop character image vertically to remove empty rows at top and bottom.

    Args:
        char_img: Character image

    Returns:
        Vertically cropped character image
    """
    # Find rows with content (dark pixels)
    row_intensities = np.mean(char_img.astype(np.float32), axis=1)
    print(f"Row intensities: {row_intensities}")
    
    content_rows = row_intensities < 200  # Rows with some dark content

    if not np.any(content_rows):
        return char_img  # No content found, return original

    # Find first and last rows with content
    content_indices = np.where(content_rows)[0]
    top_row = content_indices[0]
    bottom_row = content_indices[-1] + 1
    

    return char_img[top_row-2:bottom_row+2, :]



import numpy as np
import cv2
from typing import Tuple, Union

def remove_almost_black_borders(
    img: np.ndarray,
    proportion_threshold: float = 0.95,
    pixel_value_threshold: Optional[float] = None,
    check_columns_first: bool = True
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Remove columns (then rows) that are almost all black.

    Args:
      img: input image, shape (H,W) or (H,W,C), dtype uint8 or float32/float64.
      proportion_threshold: fraction of pixels in a column/row that must be <= pixel_value_threshold
                            to consider that column/row "black" and remove it (default 0.95).
      pixel_value_threshold:
            - if None: chosen automatically: 10 for uint8, 0.05 for float images.
            - otherwise use this numeric threshold (same scale as img).
      check_columns_first: if True remove columns first, then rows. Otherwise rows then columns.

    Returns:
      (out_img, removed_columns, removed_rows)
      - out_img: cropped image (same dtype and channels as input)
      - removed_columns: list of removed column indices (in original coordinates)
      - removed_rows: list of removed row indices (in original coordinates)

    Notes:
      - This removes ANY column/row meeting the criterion (not only border runs).
      - If you want to only remove contiguous borders at the edges, let me know and I can adapt the function.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")

    if img.size == 0:
        return img, [], []

    # copy reference to original dtype and shape
    orig_dtype = img.dtype
    h, w = img.shape[:2]

    # choose pixel threshold automatically if not provided
    if pixel_value_threshold is None:
        if np.issubdtype(orig_dtype, np.floating):
            pv_thr = 0.05
        else:
            pv_thr = 10  # for uint8 0..255
    else:
        pv_thr = float(pixel_value_threshold)

    # compute single-channel intensity for checking (grayscale float 0..1)
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if hasattr(cv2, 'cvtColor') else \
               (0.299*img[...,2] + 0.587*img[...,1] + 0.114*img[...,0])
    else:
        gray = img.copy()

    # convert gray to float in 0..1 for stable thresholding logic if input is uint8
    if np.issubdtype(gray.dtype, np.floating):
        gray_f = gray.astype(np.float32)
        if gray_f.max() > 1.5:
            gray_f = gray_f / 255.0
    else:
        gray_f = gray.astype(np.float32) / 255.0

    # convert pv_thr to 0..1 scale consistent with gray_f
    if np.issubdtype(orig_dtype, np.floating):
        thr = pv_thr
    else:
        thr = float(pv_thr) / 255.0

    # helper to find indices of columns to remove
    def _find_black_columns(gf: np.ndarray) -> List[int]:
        # gf shape (H, W)
        H, W = gf.shape
        # compute fraction of pixels <= thr per column
        # boolean array shape (H, W)
        black_mask_cols = gf <= thr
        frac_black = black_mask_cols.sum(axis=0) / float(H)
        cols = np.where(frac_black >= proportion_threshold)[0].tolist()
        return cols

    def _find_black_rows(gf: np.ndarray) -> List[int]:
        H, W = gf.shape
        black_mask_rows = gf <= thr
        frac_black = black_mask_rows.sum(axis=1) / float(W)
        rows = np.where(frac_black >= proportion_threshold)[0].tolist()
        return rows

    # operate on a working copy of gray_f to compute indices, but crop original img
    removed_cols = []
    removed_rows = []

    if check_columns_first:
        cols = _find_black_columns(gray_f)
        removed_cols = cols
        if cols:
            # create mask of columns to keep
            keep_cols = np.ones(w, dtype=bool)
            keep_cols[cols] = False
            if keep_cols.sum() == 0:
                # would remove everything -> avoid that, return original
                return img.copy(), [], []
            # apply to original image (all channels)
            if img.ndim == 2:
                img = img[:, keep_cols]
            else:
                img = img[:, keep_cols, :]
            # also update gray_f and new dims
            gray_f = gray_f[:, keep_cols]
            h, w = img.shape[:2]

    # now rows
    rows = _find_black_rows(gray_f)
    removed_rows = rows
    if rows:
        keep_rows = np.ones(h, dtype=bool)
        keep_rows[rows] = False
        if keep_rows.sum() == 0:
            return img.copy(), removed_cols, []
        if img.ndim == 2:
            img = img[keep_rows, :]
        else:
            img = img[keep_rows, :, :]
        # update gray and dims if needed
        gray_f = gray_f[keep_rows, :]
        h, w = img.shape[:2]

    # if user wanted rows first, repeat symmetrical logic
    if not check_columns_first:
        # after removing rows, remove columns
        cols = _find_black_columns(gray_f)
        removed_cols = cols
        if cols:
            keep_cols = np.ones(w, dtype=bool)
            keep_cols[cols] = False
            if keep_cols.sum() == 0:
                return img.copy(), [], removed_rows
            if img.ndim == 2:
                img = img[:, keep_cols]
            else:
                img = img[:, keep_cols, :]
            gray_f = gray_f[:, keep_cols]
            h, w = img.shape[:2]

    return img, removed_cols, removed_rows


def pad_or_crop_to_target(
    img: np.ndarray,
    target_size: Tuple[int, int] = (80,20),
    bg_color: Union[int, Tuple[int, int, int]] = 255
) -> np.ndarray:
    """
    Pads or crops an image to a fixed size without rescaling.
    Always adds white padding (not black), even for float images.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if img.ndim not in (2, 3):
        raise ValueError("img must be 2D (H,W) or 3D (H,W,C)")

    target_w, target_h = 32, 32
    h, w = img.shape[:2]
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

    # --- Center crop if larger ---
    if w > target_w or h > target_h:
        left = max((w - target_w) // 2, 0)
        top = max((h - target_h) // 2, 0)
        right = left + min(target_w, w)
        bottom = top + min(target_h, h)
        img = img[top:bottom, left:right]
        h, w = img.shape[:2]

    # --- Add symmetric white padding if smaller ---
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)
    # pad_top = max((target_h - h) // 2, 0)
    # pad_bottom = max(target_h - h - pad_top, 0)
    # pad_left = 10
    # pad_right = 10
    # pad_top = 10
    # pad_bottom = 10
    
    # pad_left = 
    # pad_right = 10
    pad_top = 0
    pad_bottom = 0
    padded = cv2.copyMakeBorder(
        img,
        top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=bg
    )

    return padded


def separate_characters_advanced(img_cell: np.ndarray,
                                intensity_threshold: float = 0.2,
                                min_char_width: int = 2,
                                debug=True, cell_path="results/cell_r12_c1.png") -> List[np.ndarray]:
    """
    Separate characters in cell image using advanced column intensity analysis.

    Args:
        img_cell: Input cell image (BGR or grayscale)
        intensity_threshold: Threshold for determining character vs separator columns
        min_char_width: Minimum width for a valid character
        debug: Whether to print debugging information

    Returns:
        List of character images
    """
    
    min_char_image_width= 6
    # Convert to grayscale if needed
    if img_cell.ndim == 3:
        gray = cv2.cvtColor(img_cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cell.copy()
    print("Gray shape is: ", gray.shape)
    gray, removed_cols, removed_rows = remove_almost_black_borders(gray)

    if debug:
        print(f"Original image shape: {gray.shape}")
        print(f"Original image intensity range: {gray.min()} - {gray.max()}")

    # Remove cell boundaries
    # gray_clean = remove_cell_boundaries(gray)
    
    # save gray_clean for debug
    random_int = random.randint(0, 1000000)
    cv2.imwrite(f"results/cell_no_boundaries_{random_int}.png", gray)
    
    print(f"Shape of current image is: {gray.shape}")
    

    if debug:
        print(f"After boundary removal, intensity range: {gray.min()} - {gray.max()}")

    # blobs = extract_blobs(gray)
    # print(f"Len of blobs is: {len(blobs)}")
    count = 2
    
    
    char_boundaries = []
    final_characters = []
    base_name = os.path.splitext(os.path.basename(cell_path))[0]
    count_blobs = 0
    # for blob in blobs:
        # print("Saving blob")
        # out_float = pad_or_crop_to_target(blob, target_size=(80,20), bg_color=1.0)
        # out_float = blob
        # cv2.imwrite(f"blobs/{base_name}_blob_{count_blobs}.png", out_float)
        # blobs_2 = extract_blobs(blob)
        # print(f"Len of blobs_2 is: {len(blobs_2)}")
        # if count == 1:
        # print(f"Blob shape is: {blob.shape}")
    column_intensities=calculate_column_intensity(gray, count)
    print(f"COLUMN INTENSITIES: {column_intensities}")
    char_boundaries, count, states = find_character_boundaries_by_columns(
            gray, column_intensities, min_char_width=2, min_gap_width=1, return_states=True, boundary = 2.5
        )
        # char_boundaries.extend(char_boundaries)
        
    print(f"Found {len(char_boundaries)} potential character regions: {char_boundaries}")
    print('PROCEED')

    character_images = []
    for i, (start_col, end_col) in enumerate(char_boundaries):
        print(f"Processing region {i}: columns {start_col}-{end_col}")

        # Extract character region
        char_width = end_col - start_col
        if char_width >= min_char_width:

            print(f"char_width: {char_width}")
            
            
            if end_col - start_col > 10:
                end_col_o, start_col_o = end_col, start_col
                added = False

                print("Character is too wide, try to crop again")
                new_crop = gray[:, start_col-1:end_col+1]
                char_boundaries, count, states = find_character_boundaries_by_columns(
                     new_crop, column_intensities, min_char_width=2, min_gap_width=1, return_states=True, boundary = 2
                )
                print(f"Found {len(char_boundaries)} potential character regions: {char_boundaries}")
                
                for i, (start_col, end_col) in enumerate(char_boundaries):
                    if end_col - start_col > 10:
                        print("Character is still too wide, giving up")
                    char_img = new_crop[:, start_col-1:end_col+1]
                    added = True
                    character_images.append(char_img)
                if not added:
                    print("ADDING ANYWAY")
                    print(f"Start col: {start_col_o}, end col: {end_col_o}")
                    char_img = gray[:, start_col_o-1:end_col_o+1]
                    character_images.append(char_img)
                    
                #     if end_col - start_col <= 10:
                #         char_img = new_crop[:, start_col-1:end_col+1]
                #         added = True
                #         character_images.append(char_img)
                #     else:
                #         print("Character is too wide, try to crop again")
                #         new_crop = gray[:, start_col-1:end_col+1]
                #         char_boundaries, count, states = find_character_boundaries_by_columns(
                #             new_crop, column_intensities, min_char_width=2, min_gap_width=1, return_states=True, boundary = 1
                #         )
                #         print(f"Found {len(char_boundaries)} potential character regions: {char_boundaries}")
                #         for i, (start_col_e, end_col_e) in enumerate(char_boundaries):
                #             if end_col_e - start_col_e <= 10:
                #                 char_img = new_crop[:, start_col_e-1:end_col_e+1]
                #                 character_images.append(char_img)
                #                 print("Added")
                #                 added = True
                # if not added:
                #     print("ADDING ANYWAY")
                #     print(f"Start col: {start_col_o}, end col: {end_col_o}")
                #     char_img = gray[:, start_col_o-1:end_col_o+1]
                #     character_images.append(char_img)
            
            else:   
            
                char_img = gray[:, start_col-1:end_col+1]
                character_images.append(char_img)
            
            

            # Crop vertically to remove empty space
            # char_img_cropped = crop_character_vertically(char_img)
            # print(f"char_img_cropped shape: {char_img_cropped.shape}")

            # Only add if the cropped image has reasonable dimensions
            # if char_img_cropped.shape[0] >= 2 and char_img_cropped.shape[1] >= 2:
            #     character_images.append(char_img_cropped)
            #     if debug:
            #         print(f"  Added character {len(character_images)-1} with size {char_img_cropped.shape}")
            # elif debug:
            #     print(f"  Rejected: cropped size {char_img_cropped.shape} too small")
        elif debug:
            print(f"  Rejected: width {char_width} < minimum {min_char_width}")
    print(f"Found {len(character_images)} characters in blob {count_blobs}")
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(cell_path))[0]

    final_characters.extend(character_images)
    # Save each character
    output_dir = "characters"
    print(f"Saving {len(character_images)} characters to {output_dir}")
    for i, char_img in enumerate(character_images):
        output_path = os.path.join(output_dir, f"{base_name}_char_{i:02d}.png")
        out_float = pad_or_crop_to_target(char_img, target_size=(80,20), bg_color=1.0)
        # out_float = char_img
        cv2.imwrite(output_path, out_float)
        print(f"Saved character {i} to {output_path} (size: {out_float.shape})")
        # print(f"Saved character {i} to {output_path} (size: {char_img.shape})")
        count += 1
    count_blobs += 1
    

    
    # Calculate column intensities
    # column_intensities = calculate_column_intensity(gray)
    # print(f"Intensities: {column_intensities}")

    # if debug:
    #     print(f"Column intensities shape: {column_intensities.shape}")
    #     print(f"Column intensities range: {column_intensities.min():.3f} - {column_intensities.max():.3f}")
    #     print(f"Mean column intensity: {column_intensities.mean():.3f}")
    #     print(f"Columns above threshold ({intensity_threshold}): {np.sum(column_intensities > intensity_threshold)}")

    return final_characters

def process_single_cell_and_save(cell_path: str, output_dir: str = "results",
                                debug: bool = True) -> None:
    """
    Process a single cell image and save separated characters.

    Args:
        cell_path: Path to cell image
        output_dir: Directory to save results
        debug: Whether to print debugging information
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(cell_path)
    if img is None:
        print(f"Error: Could not load image {cell_path}")
        return

    # img = upscale_image(img, 4)
    # img = mask_pixels(img, "results/masked.png")
    # print(f"Processing {cell_path}...")
    

    # Try different thresholds to find characters
    # thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    thresholds=[0.1]
    best_result = []
    best_threshold = 0.2
    

    for threshold in thresholds:
        characters = separate_characters_advanced(img, intensity_threshold=threshold,
                                                min_char_width=2, debug=True and threshold == 0.15, cell_path=cell_path)
        if len(characters) > len(best_result):
            best_result = characters
            best_threshold = threshold
        if debug:
            print(f"Threshold {threshold}: found {len(characters)} characters")

    characters = best_result
    print(f"Using threshold {best_threshold}, found {len(characters)} characters")

    

# -------------------------
# Main function for single cell processing
# -------------------------
def process_single_cell_main(cell_image_path: str = "cells_production/cell_r1_c1.png") -> None:
    """
    Main function to process a single cell image and separate characters.
    This is the main function requested by the user.

    Args:
        cell_image_path: Path to the cell image to process
    """
    print(f"Processing cell image: {cell_image_path}")
    print("="*60)

    # Process the cell and save results
    process_single_cell_and_save(cell_image_path, debug=True)

    print("\nCharacter separation completed!")
    print("Results saved in 'results/' folder")

# -------------------------
# Example usage
# -------------------------



if __name__ == "__main__":
    # Test with a single cell as requested
    count = 0
    working_directory = "words_production"
    for filename in os.listdir(working_directory):
        # if count > 10:
        #     break
        if filename.endswith('.png'):
            count += 1
            cell_path = os.path.join(working_directory, filename)
            process_single_cell_main(cell_path)
    
    # path="words_production/cell_r5_c1_blob_1_word_7.png"
    # process_single_cell_and_save(path)
    # blobs = extract_blobs(img)
    # print(len(blobs))
    
    # calculate_column_intensity_top_bottom(img)

    # Optionally test with multiple images for comparison
    test_multiple = False
    if test_multiple:
        test_images = [
            "cells_production/cell_r2_c1.png",  # This has good content
            "cells_production/cell_r1_c1.png",   # This has content across many columns
            "cells_production/cell_r10_c5.png",  # This has good content
            "cells_production/cell_r0_c3.png"    # This has minimal content
        ]

        for img_path in test_images:
            print(f"\n{'='*60}")
            print(f"Testing: {img_path}")
            print('='*60)

            # Process single cell and save results
            process_single_cell_and_save(img_path, debug=True)

            # Also show the old segmentation for comparison
            img = cv2.imread(img_path)
            if img is not None:
                entries = segment_cell(img)
                print(f"Old method found {len(entries)} candidate symbols.")

            print()  # Empty line for readability







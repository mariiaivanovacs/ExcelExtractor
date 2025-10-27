import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import string
import csv
import random
import math
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# Create directories if they don't exist
os.makedirs('experiment/synth', exist_ok=True)




# Function to find center of mass (as mentioned in the requirements)
from calculate import show_central_mass


import cv2, random, numpy as np

def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    """
    Simulate low-res, pixelated distortion for white text on black background.

    Args:
        np_img: Input grayscale image (white text on black background).
        min_scale, max_scale: Downsampling scale range.
    Returns:
        np.ndarray: Distorted grayscale image.
    """
    max_shift = 3
    blur_chance = 0.5

    h, w = np_img.shape[:2]

    # Step 1: Randomly scale down & up (pixelation)
    scale = random.uniform(0.8, 0.83)
    small = cv2.resize(np_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    
    # Set a threshold — e.g., remove all pixels darker than 50
    threshold_value = 50
    _, img = cv2.threshold(up, threshold_value, 255, cv2.THRESH_TOZERO)

    # Step 2: Apply small random affine transform (to jitter or shift)
    # dx = random.randint(-max_shift, max_shift)
    # dy = random.randint(-max_shift, max_shift)
    # M = np.float32([[1, 0, dx], [0, 1, dy]])
    # up = cv2.warpAffine(up, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < blur_chance:
        
        # Step 3: Lightly “fade” bright regions (since text is white)
        up = up.astype(np.float32)
        factor = random.uniform(0.2, 0.3)  # how much to dim bright areas
        mask = up > 155  # only bright pixels (the text)
        up[mask] = up[mask] - (up[mask] - 155) * factor  # pull toward gray
        up = np.clip(up, 0, 255).astype(np.uint8)

    # Step 4: Optional small blur to soften edges
    
    up = cv2.GaussianBlur(up, (5, 5), sigmaX=random.uniform(0.7, 1.0))

    return up


# Function to extract features
# def extract_features(image):
#     # Convert to grayscale
#     if len(image.shape) > 2:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()
    
#     height, width = gray.shape
    
#     # 1. First moment (mean)
#     first_moment = np.mean(gray)
    
#     # 2. Second moment (variance)
#     second_moment = np.var(gray)
    
#     # 3. Magnitude (Frobenius norm)
#     magnitude = np.linalg.norm(gray)
    
#     # 4. Edge density
#     edges = cv2.Canny(gray, 100, 200)
#     edge_density = np.sum(edges > 0) / (height * width)
    
#     # 5. Columns intensity
#     col_intensity = np.sum(gray, axis=0) / height
    
#     # 6. Rows intensity
#     row_intensity = np.sum(gray, axis=1) / width
    
#     # 7. HOG (Histogram of Oriented Gradients)
#     # Using a simplified HOG implementation
#     gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     mag, ang = cv2.cartToPolar(gx, gy)
    
#     # Simple HOG: 9 orientation bins
#     hist = np.zeros(9)
#     for i in range(height):
#         for j in range(width):
#             if mag[i,j] > 0:
#                 bin_idx = int(ang[i,j] * 9 / (2 * np.pi)) % 9
#                 hist[bin_idx] += mag[i,j]
#     hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist
    
#     # 8. Hu moments (7 invariant moments)
#     moments = cv2.moments(gray)
#     hu_moments = cv2.HuMoments(moments).flatten()
    
#     # We won't implement full Zernike moments as they're complex
#     # but we'll use Hu moments as they're related
    
#     # 9. Symmetry (horizontal and vertical)
#     h_flipped = np.fliplr(gray)
#     v_flipped = np.flipud(gray)
#     h_symmetry = 1 - np.sum(np.abs(gray - h_flipped)) / (255 * width * height)
#     v_symmetry = 1 - np.sum(np.abs(gray - v_flipped)) / (255 * width * height)
#     symmetry = (h_symmetry + v_symmetry) / 2
    
#     # 10. Aspect ratio (for the non-zero region)
#     # Find bounding box of non-zero pixels
#     non_zero = cv2.findNonZero(gray)
#     if non_zero is not None:
#         x, y, w, h = cv2.boundingRect(non_zero)
#         aspect_ratio = w / h if h > 0 else 1.0
#     else:
#         aspect_ratio = 1.0
    
#     # 11. Projection variance
#     col_projection_var = np.var(col_intensity)
#     row_projection_var = np.var(row_intensity)
#     projection_variance = (col_projection_var + row_projection_var) / 2
    
#     # 12. L1 Normalized of gradient map
#     gradient_magnitude = np.abs(gx) + np.abs(gy)  # L1 norm of gradient
#     l1_normalized_gradient = np.sum(gradient_magnitude) / (width * height)
    
#     # Combine all features
#     features = {
#         'first_moment': first_moment,
#         'second_moment': second_moment,
#         'magnitude': magnitude,
#         'edge_density': edge_density,
#         # 'column_intensity': col_intensity.tolist(),  # Converting numpy array to list for CSV storage
#         # 'row_intensity': row_intensity.tolist(),
#         'hog': hist.tolist(),
#         'hu_moments': hu_moments.tolist(),
#         'symmetry': symmetry,
#         'aspect_ratio': aspect_ratio,
#         'projection_variance': projection_variance,
#         'l1_normalized_gradient': l1_normalized_gradient
#     }
    
#     return features


def compute_left_right_columns(img_arr, thresh=250):
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
        start = 1
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



def extract_features(image):
    # Convert to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy().astype(np.uint8)
    height, width = gray.shape
    area = height * width

    # 1. First moment (mean) and 2. Second moment (variance)
    first_moment = float(np.mean(gray))
    second_moment = float(np.var(gray))

    # 3. Magnitude (Frobenius norm)
    magnitude = float(np.linalg.norm(gray))

    # 4. Canny edge & density
    canny = cv2.Canny(gray, 100, 200)
    canny_edge_density = float(np.sum(canny > 0) / area)

    # 5/6. Column / Row intensity projections
    col_intensity = np.sum(gray, axis=0) / float(height)
    row_intensity = np.sum(gray, axis=1) / float(width)

    # vertical intensity variance (requested)
    vertical_intensity_variance = float(np.var(col_intensity))

    # 7. Sobel gradients (gx, gy), magnitude and angle
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.hypot(gx, gy)  # equivalent to cartToPolar magnitude
    sobel_mean_magnitude = float(np.mean(sobel_mag))
    sobel_std_magnitude = float(np.std(sobel_mag))

    # sobel_edge_density: fraction of pixels with sobel magnitude > mean
    sobel_edge_density = float(np.sum(sobel_mag > sobel_mean_magnitude) / area)

    # compute angle map (0..2pi)
    ang = (np.arctan2(gy, gx) + 2*np.pi) % (2*np.pi)

    # edge_direction_ratio: vertical_mag / horizontal_mag
    # define vertical as angles near pi/2, 3pi/2; horizontal near 0, pi
    vertical_mask = ((ang > np.pi/4) & (ang < 3*np.pi/4)) | ((ang > 5*np.pi/4) & (ang < 7*np.pi/4))
    horizontal_mask = ~vertical_mask
    vertical_mag = np.sum(sobel_mag[vertical_mask])
    horizontal_mag = np.sum(sobel_mag[horizontal_mask])
    edge_direction_ratio = float(vertical_mag / (horizontal_mag + 1e-12))

    # 8. Simple HOG: 9 orientation bins (using sobel magnitudes & angles)
    hog_bins = 9
    hog_hist = np.zeros(hog_bins, dtype=float)
    # accumulate magnitudes into bins
    bin_idx = np.floor(ang * hog_bins / (2*np.pi)).astype(int) % hog_bins
    for b in range(hog_bins):
        hog_hist[b] = np.sum(sobel_mag[bin_idx == b])
    if hog_hist.sum() > 0:
        hog_hist = hog_hist / hog_hist.sum()
    hog_mean = float(np.mean(hog_hist))
    hog_std = float(np.std(hog_hist))
    hog_max = float(np.max(hog_hist))
    hog_energy = float(np.sum(hog_hist ** 2))

    # 9. Hu moments (7)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = hu_moments.astype(float).tolist()

    # 10. Symmetry (horizontal and vertical)
    h_flipped = np.fliplr(gray)
    v_flipped = np.flipud(gray)
    # normalize by max possible difference (255 * area)
    h_symmetry = 1.0 - (np.sum(np.abs(gray - h_flipped)) / (255.0 * area))
    v_symmetry = 1.0 - (np.sum(np.abs(gray - v_flipped)) / (255.0 * area))
    symmetry = float((h_symmetry + v_symmetry) / 2.0)

    # 11. Aspect ratio (for the non-zero region)
    non_zero = cv2.findNonZero(gray)
    if non_zero is not None:
        x, y, w, h = cv2.boundingRect(non_zero)
        aspect_ratio = float(w / h) if h > 0 else 1.0
    else:
        aspect_ratio = 1.0

    # 12. Projection variance
    col_projection_var = float(np.var(col_intensity))
    row_projection_var = float(np.var(row_intensity))
    projection_variance = float((col_projection_var + row_projection_var) / 2.0)

    # 13. L1 normalized gradient (sum of absolute gx,gy)
    gradient_magnitude = np.abs(gx) + np.abs(gy)
    l1_normalized_gradient = float(np.sum(gradient_magnitude) / area)

    # ====== NEW ADVANCED FEATURES ======

    # 14. Projection correlation - correlation between row and column projections
    # Resize projections to same length for correlation calculation
    min_len = min(len(row_intensity), len(col_intensity))
    row_resized = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(row_intensity)), row_intensity)
    col_resized = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(col_intensity)), col_intensity)

    if np.std(row_resized) > 0 and np.std(col_resized) > 0:
        projection_correlation = float(np.corrcoef(row_resized, col_resized)[0, 1])
    else:
        projection_correlation = 0.0

    # 15. Energy ratio - ratio of energy in upper vs lower half
    upper_half = gray[:height//2, :]
    lower_half = gray[height//2:, :]
    upper_energy = float(np.sum(upper_half.astype(np.float64) ** 2))
    lower_energy = float(np.sum(lower_half.astype(np.float64) ** 2))
    energy_ratio = float(upper_energy / (lower_energy + 1e-12))

    # 16. Quadrant energy ratio - energy distribution across 4 quadrants
    h_mid, w_mid = height // 2, width // 2
    q1 = gray[:h_mid, :w_mid]      # top-left
    q2 = gray[:h_mid, w_mid:]      # top-right
    q3 = gray[h_mid:, :w_mid]      # bottom-left
    q4 = gray[h_mid:, w_mid:]      # bottom-right

    q1_energy = float(np.sum(q1.astype(np.float64) ** 2))
    q2_energy = float(np.sum(q2.astype(np.float64) ** 2))
    q3_energy = float(np.sum(q3.astype(np.float64) ** 2))
    q4_energy = float(np.sum(q4.astype(np.float64) ** 2))

    total_quad_energy = q1_energy + q2_energy + q3_energy + q4_energy
    if total_quad_energy > 0:
        q1_ratio = float(q1_energy / total_quad_energy)
        q2_ratio = float(q2_energy / total_quad_energy)
        q3_ratio = float(q3_energy / total_quad_energy)
        q4_ratio = float(q4_energy / total_quad_energy)
    else:
        q1_ratio = q2_ratio = q3_ratio = q4_ratio = 0.25

    # Quadrant energy variance (how uneven the distribution is)
    quad_energies = [q1_ratio, q2_ratio, q3_ratio, q4_ratio]
    quadrant_energy_variance = float(np.var(quad_energies))

    # 17. Row variance vs column variance ratio
    row_variance_ratio = float(row_projection_var / (col_projection_var + 1e-12))

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

    # ====== Binary mask & contours / topology ======
    # Use Otsu to binarize, then ensure foreground (character) is white (255)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If there are more white pixels than black, invert so that character becomes white (smaller region)
    white_count = int(np.sum(binary == 255))
    black_count = int(np.sum(binary == 0))
    if white_count > black_count:
        binary = cv2.bitwise_not(binary)

    # find contours (tree to get parent-child relationships)
    contours_info = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    hierarchy = contours_info[1] if len(contours_info) == 3 else (None if len(contours_info) == 2 else contours_info[2])
    num_contours = len(contours)

    # contour solidity: area / hull_area (average over contours with non-zero hull)
    solidities = []
    contour_areas = []
    contour_perimeters = []
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a <= 0:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(a / hull_area) if hull_area > 0 else 0.0
        solidities.append(solidity)
        contour_areas.append(a)
        contour_perimeters.append(cv2.arcLength(cnt, True))
    avg_contour_solidity = float(np.mean(solidities)) if len(solidities) > 0 else 0.0
    avg_contour_area = float(np.mean(contour_areas)) if len(contour_areas) > 0 else 0.0

    # Using hierarchy to compute holes & external contour count
    # hole_count = 0
    # external_count = 0
    # if hierarchy is not None and len(contours) > 0:
    #     # hierarchy shape: (1, N, 4)
    #     h = np.array(hierarchy).reshape(-1, 4)
    #     parents = h[:, 3]  # parent index for each contour
    #     external_count = int(np.sum(parents == -1))
    #     hole_count = int(num_contours - external_count)
    # else:
    #     # fallback: assume no holes if no contours or no hierarchy
    #     hole_count = 0
    #     external_count = 0

    # Euler number = #objects - #holes  (objects ~ external_count)
    # euler_number = float(external_count - hole_count)

    # compactness: (4*pi*area) / perimeter^2 for the largest contour (1 for circle)
    compactness = 0.0
    if len(contour_areas) > 0:
        largest_idx = int(np.argmax(contour_areas))
        A = contour_areas[largest_idx]
        P = contour_perimeters[largest_idx] if len(contour_perimeters) > largest_idx else 0.0
        if P > 0 and A > 0:
            compactness = float((4.0 * np.pi * A) / (P * P))
        else:
            compactness = 0.0

    # Connected components (cc_count)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((binary == 255).astype(np.uint8), connectivity=8)
    # Note: num_labels includes background label 0
    cc_count = int(num_labels - 1)
    # avg_cc_area
    if cc_count > 0:
        cc_areas = stats[1:, cv2.CC_STAT_AREA]
        avg_cc_area = float(np.mean(cc_areas))
        cc_area_std = float(np.std(cc_areas))
    else:
        avg_cc_area = 0.0
        cc_area_std = 0.0
        
        

    # b = (img>0).astype(np.uint8)
    # cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # if not cnts: return {"curv_mean":0,"curv_std":0,"curv_hist":[]}
    # c = max(cnts, key=cv2.contourArea).squeeze()
    # # compute tangent vectors and angle between successive segments
    # diffs = np.diff(c, axis=0)
    # # wrap last to first
    # diffs = np.vstack([diffs, c[0]-c[-1]])
    # angles = np.arctan2(diffs[:,1], diffs[:,0])
    # # curvature = angle change between successive tangents
    # dangles = np.diff(np.unwrap(angles))
    # curv = np.abs(dangles)
    # hist, _ = np.histogram(curv, bins=8, range=(0, np.pi))
    # return {"curv_mean":float(curv.mean() if curv.size else 0),
    #         "curv_std":float(curv.std() if curv.size else 0),
    #         "curv_hist":hist.tolist()}

    # Small morphology to close broken loops
    
    # Adaptive threshold gives stability for light/dark variations
    b = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # invert so digits are white
        25, 10
    )

    # Morphological closing to fix broken loops (like 0, 6, 8, 9)
    kernel = np.ones((3, 3), np.uint8)
    
    
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Optional: small opening to remove single pixel noise
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Optional dilation — helps reconnect thin strokes in 6/8
    kernel_dilate = np.ones((2, 2), np.uint8)
    b = cv2.dilate(b, kernel_dilate, iterations=1)

    # Optional erosion to recover original thickness
    b = cv2.erode(b, kernel_dilate, iterations=1)
    
    # find contours of foreground, then count holes using connected components on inverted ROI
    contours, _ = cv2.findContours(b.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # RETR_CCOMP gives parent/child relationships; count child contours (holes)
    hole_count = 0
    # contours with parent >=0 are holes (in RETR_CCOMP)
    # OpenCV's return hierarchy needed:
    # ret, contours, hierarchy = cv2.findContours(...)
    # hierarchy[0][i][3] = parent
    # If using cv2.findContours with hierarchy:
    ret = cv2.findContours(b.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 3:
        _, contours, hierarchy = ret
    else:
        contours, hierarchy = ret
    if hierarchy is not None:
        hier = hierarchy[0]
        h = np.array(hierarchy).reshape(-1, 4)
        parents = h[:, 3]  # par
        # holes are contours whose parent != -1
        hole_count = int((hier[:,3] != -1).sum())
        external_count = int(np.sum(parents == -1))

        
    euler_number = float(external_count - hole_count)

        
    arr = np.array(gray)  # 2D array
    left_idx, left_sums, right_idx, right_sums, bbox = compute_left_right_columns(arr)
    
    
    top_sums, bottom_sums, left_sums, right_sums, top_idx, bot_idx = compute_top_bottom_rows(arr, left_idx, right_idx, thresh=250,)
    left, right, top, bottom = left_idx[0], right_idx[1], top_idx[0], bot_idx[1]
    avgs, boxes = split_character_into_8_areas(arr, left, right, top, bottom)


    
            # print("top rows:", top_idx, "sums:", top_sums)
    
    # ===== Assemble features =====
    features = {
        # 'first_moment': first_moment,
        'second_moment': second_moment,
        # 'magnitude': magnitude,

        # edges
        # 'canny_edge_density': canny_edge_density,
        # 'sobel_edge_density': sobel_edge_density,
        'sobel_mean_magnitude': sobel_mean_magnitude,
        'sobel_std_magnitude': sobel_std_magnitude,
        # 'edge_direction_ratio': edge_direction_ratio,

        # HOG summaries
        # 'hog_mean': hog_mean,
        # 'hog_std': hog_std,
        # 'hog_max': hog_max,
        # 'hog_energy': hog_energy,

        # Hu moments (7)
        # 'hu_moments': hu_moments,

        # 'symmetry': symmetry,
        'aspect_ratio': aspect_ratio,
        # 'projection_variance': projection_variance,
        'vertical_intensity_variance': vertical_intensity_variance,
        # 'l1_normalized_gradient': l1_normalized_gradient,

        # NEW ADVANCED FEATURES
        # 'projection_correlation': projection_correlation,
        # 'energy_ratio': energy_ratio,
        # 'q1_energy_ratio': q1_ratio,
        # 'q2_energy_ratio': q2_ratio,
        # 'q3_energy_ratio': q3_ratio,
        # 'q4_energy_ratio': q4_ratio,
        # 'quadrant_energy_variance': quadrant_energy_variance,
        # 'row_variance_ratio': row_variance_ratio,
        # 'row_entropy': row_entropy,
        # 'col_entropy': col_entropy,
        # 'projection_entropy_ratio': projection_entropy_ratio,

        # contour / topology
        'num_contours': int(num_contours),
        'hole_count': int(hole_count),
        'avg_euler_number': euler_number,
        'avg_contour_solidity': avg_contour_solidity,
        'avg_contour_area': avg_contour_area,
        'compactness': compactness,

        # connected components
        'cc_count': cc_count,
        'avg_cc_area': avg_cc_area,
        'cc_area_std': cc_area_std,
        
        "left_sum": sum(left_sums),
        "right_sum": sum(right_sums),
        "top_sum": sum(top_sums),
        "bottom_sum": sum(bottom_sums),
        
        "avgs": avgs,
        
        

        # # keep the full hog histogram too (optional)
        # 'hog_hist': hog_hist.tolist()
    }

    return features


    
def build_characters(extra_tokens=None, target_total_tokens=200):
    """
    Build an inclusive characters/tokens list:
      - ASCII letters + digits
      - Cyrillic uppercase/lowercase (incl. Ё/ё)
      - two-digit numbers "00".."99"
      - dot-prefixed decimals ".0".." .9"
      - dot-suffixed numbers "1.".."20."
      - a set of common Russian short tokens (editable)
      - optional auto-generated bigrams to reach desired token count

    Returns:
        List[str] characters  # unique tokens
    """
    chars = []

    # 1) Basic ASCII letters & digits
    # chars.extend(list(string.ascii_letters))   # A-Z a-z
    # chars.extend(list(string.digits))          # 0-9

    # 2) Cyrillic letters (Russian alphabet)
    # Uppercase А..Я : U+0410..U+042F ; include Ё (U+0401)
    # Lowercase а..я : U+0430..U+044F ; include ё (U+0451)
    # cyr_upper = [chr(cp) for cp in range(0x0410, 0x042F + 1)]
    # cyr_lower = [chr(cp) for cp in range(0x0430, 0x044F + 1)]
    # # add Ё/ё explicitly
    # cyr_extra = [chr(0x0401), chr(0x0451)]
    # chars.extend(cyr_upper + cyr_extra + cyr_lower)

    # # 3) Two-digit strings "00".."99"
    # two_digits = [f"{i}{i}" for i in range(10)]  # "00", "11", ..., "99"

    # chars.extend(two_digits)

    # 4) Dot-prefixed decimals and single-digit dot suffixes
    dot_prefix = [f".{d}" for d in range(10)]         # .0 .. .9
    dot_suffix = [f"{d}." for d in range(1, 21)]      # 1. .. 20.
    chars.extend(dot_prefix + dot_suffix)

    # # 5) Some other numeric patterns you mentioned
    # e.g. "0.0", "1.0", "2.5" etc. (optional set)
    decimals = [f"{i}.0" for i in range(10)]  # 0.0 .. 9.0
    chars.extend(decimals)

    # 6) A curated list of common Russian short tokens/abbreviations (editable)
    # Add your own words here (e.g., 'оп','аб','руб', etc.)
    common_russian = [
        "оп", "аб", "руб", "коп", "г", "ул", "д", "кв", "лит", "стр",
        "т.", "см", "м", "км", "ч", "мин", "сек", "год", "месяц", "день",
        "и", "в", "не", "на", "с", "по", "за", "от", "до", "при",
        "он", "она", "они", "мы", "ты", "я", "что", "это", "как", "для",
        "что", "так", "ещё", "у", "задача", "пример", "руб.", "тг", "евро",
        # add more tokens you specifically need...
    ]
    chars.extend(common_russian)

    # 7) Add any extra tokens provided by user
    if extra_tokens:
        chars.extend(list(extra_tokens))

    # 8) If you want around N tokens (target_total_tokens), auto-generate short Cyrillic bigrams
    #    using a small set of frequent letters to get to the target size (non-offensive)
    # if len(chars) < target_total_tokens:
    #     # Use frequent Russian letters to create useful bigrams/trigrams
    #     freq_letters = list("остеинрлмвяупкгш")  # common Russian letters (approx)
    #     i = 0
    #     # generate two-letter combinations until we reach the goal
    #     for a in freq_letters:
    #         for b in freq_letters:
    #             token = a + b
    #             chars.append(token)
    #             i += 1
    #             if len(chars) >= target_total_tokens:
    #                 break
    #         if len(chars) >= target_total_tokens:
    #             break

    # 9) Make unique and stable order (optionally sort)
    unique_chars = []
    seen = set()
    for t in chars:
        if t not in seen:
            unique_chars.append(t)
            seen.add(t)

    return unique_chars

# Example usage:
# characters = build_characters(extra_tokens=["оп", "аб", "руб"], target_total_tokens=200)
# characters = list(string.digits)  # A-Z, a-z, 0-9
image_size = 32
font_size = 14
results = []
# characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".0", ".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9", "0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

fonts = []
# Try to use a simple font
font = ImageFont.truetype("/Users/mariaivanova/Library/Fonts/calibri-regular.ttf", font_size)
fonts.append(font)
fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/Lucida Console Regular.ttf", 14))
fonts.append(font)
font=ImageFont.truetype("/Library/Fonts/adelle-sans-regular.otf", font_size)
fonts.append(font)



# num_variants = 10 # how many images per character

# for char in characters:
#     for i in range(num_variants):
#         font = random.choice(fonts)

#         # Create a blank grayscale image (black background)
#         img = Image.new('L', (image_size, image_size), color=0)
#         draw = ImageDraw.Draw(img)

#         # Compute centered text position
#         text_width, text_height = draw.textbbox((0, 0), char, font=font)[2:4]
#         position = ((image_size - text_width) // 2, (image_size - text_height) // 2)


#         # Draw white text
#         draw.text(position, char, fill=255, font=font)

#         # Convert to NumPy array
#         np_img = np.array(img)

#         # Apply random distortion
#         distorted = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)
#         # convert from black pixels to white pixels
#         distorted = cv2.bitwise_not(distorted)
        
#         # save for debug
#         # cv2.imwrite(f"experiment/debug_out/{char}_{i}.png", distorted)

#         # Convert to OpenCV format (already NumPy, so just ensure dtype)
#         img_cv = distorted.astype(np.uint8)

#         # Compute center of mass
#         mass, (cx, cy) = show_central_mass(img_cv)

#         # Extract features
#         features = extract_features(img_cv)

#         # Store results (keep pseudo path name for clarity)
#         result = {
#             'character': char,
#             'variant': i,
#             'mass': mass,
#             'center_x': cx,
#             'center_y': cy,
#             **features
#         }
#         results.append(result)

#     print(f"Processed: {char}")


working_directory = "tests"
count = 0
for filename in os.listdir(working_directory):
    if count == 20:
        break
    elif filename.endswith('.png'):
        img_path = os.path.join(working_directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Convert to NumPy array
        np_img = np.array(img)

        # ✅ Invert colors: white → black, black → white
        # np_img = cv2.bitwise_not(np_img)
        
        
        
        # store for debug
        # cv2.imwrite(f"experiment/debug_out/{filename}", np_img)


        # Apply random distortion
        # distorted = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)

        # Convert to OpenCV format (already NumPy, so just ensure dtype)
        img_cv = np_img.astype(np.uint8)

        # Compute center of mass
        mass, (cx, cy) = show_central_mass(img_cv)

        # Extract features
        features = extract_features(img_cv)

        # Store results (keep pseudo path name for clarity)
        result = {
            'character': filename,
            'variant': count,
            'mass': mass,
            'center_x': cx,
            'center_y': cy,
            **features
        }
        results.append(result)
        count += 1

    print(f"Processed: {filename}")



# Write results to CSV
csv_path = "experiment/test_features.csv"
with open(csv_path, 'w', newline='') as csvfile:
    # Get all the field names
    fieldnames = list(results[0].keys())
    print(f"Fieldnames: {fieldnames}")
    
    # apend hu_moments_1, hu_moments_2, ... to fieldnames
    for i in range(8):
        fieldnames.append(f'avg_{i+1}')
    
    # # Special handling for array fields
    # array_fields = ['column_intensity', 'row_intensity', 'hog', 'hu_moments']
    # for field in array_fields:
    #     fieldnames.remove(field)
    
    # Add individual columns for array elements
    # for field in array_fields:
    #     if field == 'column_intensity':
    #         for i in range(image_size):
    #             fieldnames.append(f'col_intensity_{i}')
    #     elif field == 'row_intensity':
    #         for i in range(image_size):
    #             fieldnames.append(f'row_intensity_{i}')
    #     elif field == 'hog':
    #         for i in range(9):  # 9 orientation bins
    #             fieldnames.append(f'hog_bin_{i}')
    #     elif field == 'hu_moments':
    #         for i in range(7):  # 7 Hu moments
    #             fieldnames.append(f'hu_moment_{i}')
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for result in results:
        row = {k: v for k, v in result.items()}  # Copy feature dict

        # If Hu moments exist and are stored as a list
        if "avgs" in row and isinstance(row["avgs"], (list, np.ndarray)):
            for i, val in enumerate(row["avgs"], start=1):
                row[f"avg_{i}"] = float(val)
            del row["avgs"]  # remove the original list to avoid issues

        
        
        # # Flatten array fields
        # for i, val in enumerate(result['column_intensity']):
        #     row[f'col_intensity_{i}'] = val
        
        # for i, val in enumerate(result['row_intensity']):
        #     row[f'row_intensity_{i}'] = val
        
        # for i, val in enumerate(result['hog']):
        #     row[f'hog_bin_{i}'] = val
        
        # for i, val in enumerate(result['hu_moments']):
        #     row[f'hu_moment_{i}'] = val
        
        writer.writerow(row)

print(f"Features extracted and saved to {csv_path}")




"""
make sense:
second moment
sobel_mean_magnitude
sobel_std_magnitude
vertical_mag 
horizontal_mag

"""


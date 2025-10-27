"""
connected_component_graph.py

Usage:
    python connected_component_graph.py /path/to/image.png

Dependencies:
    pip install opencv-python numpy matplotlib networkx

This script:
 - binarizes and optionally denoises the input image
 - finds connected components
 - builds a graph linking nearby, horizontally-aligned blobs
 - extracts clusters (words) and visualizes them
"""

import sys
import cv2
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
def remove_cell_boundaries(img: np.ndarray, boundary_thickness: int = 2) -> np.ndarray:
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

def preprocess(img_gray: np.ndarray, do_blur: bool = True) -> np.ndarray:
    """Binarize image with adaptive thresholding and optional denoising."""
    if do_blur:
        img_blur = cv2.medianBlur(img_gray, 3)
    else:
        img_blur = img_gray

    # Adaptive threshold (good for uneven illumination)
    bin_img = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=12
    )
    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    return bin_img


def extract_components(bin_img: np.ndarray, min_area: int = 10):
    """
    Returns:
      components: list of dicts with keys ['label', 'x','y','w','h','area','cx','cy']
      labels_img: indexed label image
    """
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    components = []
    for i in range(1, num_labels):  # skip background label 0
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        components.append({
            'label': i,
            'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'area': int(area),
            'cx': float(cx), 'cy': float(cy)
        })
    return components, labels_img


def build_graph(components: List[Dict],
                v_overlap_thresh: float = 0.45,
                gap_multiplier: float = 1.5):
    """
    Build an undirected graph connecting blobs that likely belong to the same word.

    Rules (defaults):
      - vertical overlap (intersection height / min(h1, h2)) >= v_overlap_thresh
      - horizontal gap between blobs <= gap_multiplier * median_width

    Returns networkx.Graph()
    """
    G = nx.Graph()
    if not components:
        return G

    widths = np.array([c['w'] for c in components])
    median_w = float(np.median(widths))
    # Fallback if median is zero
    if median_w < 1.0:
        median_w = 10.0

    for i, c in enumerate(components):
        G.add_node(i)

    # Precompute rectangles for speed
    rects = [(c['x'], c['y'], c['w'], c['h'], c['cx'], c['cy']) for c in components]

    n = len(components)
    for i in range(n):
        x1, y1, w1, h1, cx1, cy1 = rects[i]
        for j in range(i+1, n):
            x2, y2, w2, h2, cx2, cy2 = rects[j]

            # vertical intersection
            y_top = max(y1, y2)
            y_bot = min(y1 + h1, y2 + h2)
            inter_h = max(0, y_bot - y_top)
            v_overlap = inter_h / min(h1, h2) if min(h1, h2) > 0 else 0.0

            # horizontal gap (positive means a gap, negative/zero means overlap)
            if x1 <= x2:
                gap = x2 - (x1 + w1)
            else:
                gap = x1 - (x2 + w2)

            # Connection rule:
            # - vertical overlap high enough (same line)
            # - and gap not too wide relative to median width
            if v_overlap >= v_overlap_thresh and gap <= (gap_multiplier * median_w):
                G.add_edge(i, j)

    return G


def clusters_to_word_boxes(components: List[Dict], clusters: List[List[int]]) -> List[Tuple[int,int,int,int]]:
    """From list of clusters (indices) produce bounding boxes (x,y,w,h)."""
    boxes = []
    for comp in clusters:
        xs = [components[idx]['x'] for idx in comp]
        ys = [components[idx]['y'] for idx in comp]
        ws = [components[idx]['w'] for idx in comp]
        hs = [components[idx]['h'] for idx in comp]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max([xs[k] + ws[k_idx] for k_idx, k in enumerate(range(len(xs)))])  # careful compute
        # simpler computation instead:
        x_max = max([components[idx]['x'] + components[idx]['w'] for idx in comp])
        y_max = max([components[idx]['y'] + components[idx]['h'] for idx in comp])
        boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    return boxes


def visualize_clusters(img_bgr: np.ndarray, components: List[Dict], clusters: List[List[int]],
                       out_path: str = "clusters_vis.png"):
    """Draw colored bounding boxes for each cluster (word)."""
    canvas = img_bgr.copy()
    rng = np.random.default_rng(12345)
    colors = [tuple(int(c) for c in (rng.integers(0,255), rng.integers(0,255), rng.integers(0,255))) for _ in clusters]

    for ci, comp in enumerate(clusters):
        color = colors[ci]
        for idx in comp:
            c = components[idx]
            x, y, w, h = c['x'], c['y'], c['w'], c['h']
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        # draw cluster bounding box
        xs = [components[idx]['x'] for idx in comp]
        ys = [components[idx]['y'] for idx in comp]
        x_min, y_min = min(xs), min(ys)
        x_max = max([components[idx]['x'] + components[idx]['w'] for idx in comp])
        y_max = max([components[idx]['y'] + components[idx]['h'] for idx in comp])
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)
        # optional label
        cv2.putText(canvas, f"{ci}", (x_min, max(y_min-3,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f"[+] saved visualization to: {out_path}")


def extract_word_images(img_gray: np.ndarray, word_boxes: List[Tuple[int,int,int,int]], padding: int = 2):
    """Return list of cropped word images (grayscale)"""
    word_images = []
    H, W = img_gray.shape[:2]
    for (x, y, w, h) in word_boxes:
        xa = max(0, x - padding)
        ya = max(0, y - padding)
        xb = min(W, x + w + padding)
        yb = min(H, y + h + padding)
        word_images.append(img_gray[ya:yb, xa:xb])
    return word_images


def extract_blobs(img):
    img = img
    # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bin_img = preprocess(img, do_blur=True)

    components, labels_img = extract_components(bin_img, min_area=15)
    print(f"[+] found {len(components)} connected components (after filtering)")

    # Build graph and cluster
    G = build_graph(components, v_overlap_thresh=0.45, gap_multiplier=1.5)
    clusters = list(nx.connected_components(G))
    clusters = [sorted(list(c)) for c in clusters if len(c) > 0]
    print(f"[+] formed {len(clusters)} clusters (potential words)")

    # Convert clusters to bounding boxes for words
    word_boxes = clusters_to_word_boxes(components, clusters)

    # Visualize
    visualize_clusters(img, components, clusters, out_path="clusters_vis.png")

    # Optionally save each word crop
    word_images = extract_word_images(img, word_boxes, padding=3)
    # for i, wimg in enumerate(word_images):
    #     cv2.imwrite(f"results/blob_{i:03d}.png", wimg)

    print(f"[+] saved {len(word_images)} word crops as word_###.png")
    return word_images

def preprocess(img, do_blur=True):
    # Convert to grayscale if needed
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    # Ensure uint8 type
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).astype(np.uint8)

    if do_blur:
        img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    else:
        img_blur = img_gray

    # Adaptive threshold (requires uint8 grayscale)
    bin_img = cv2.adaptiveThreshold(
        img_blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=12
    )

    return bin_img


def calculate_column_intensity(img: np.ndarray, number=1, row_calc=False) -> np.ndarray:
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
    if row_calc:
        content_cols_normal = np.sum(column_intensities > (1.0 - threshold)) # Dark content
    else:
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

    output_path = f"results/inspect_{number}.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

    return column_intensities

def calculate_row_intensity(img: np.ndarray, number: int = 1) -> np.ndarray:
    """
    Calculate average intensity for each row of pixels (normalized 0..1).
    Dark ink -> higher values (inverted).

    Args:
        img: input image (H,W) grayscale uint8 or color BGR uint8 (or float in [0,1]).
        number: index used for diagnostic filename.

    Returns:
        row_intensities: 1D np.ndarray of length H with values in [0,1] (higher -> darker content).
    """
    # Ensure grayscale float in 0..1
    if img is None:
        raise ValueError("img is None")

    # If color, convert to grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Normalize to 0..1 float
    if np.issubdtype(gray.dtype, np.floating):
        gray_f = gray.astype(np.float32)
        if gray_f.max() > 1.5:  # floats in 0..255
            gray_f = gray_f / 255.0
    else:
        gray_f = gray.astype(np.float32) / 255.0
    gray_f = np.clip(gray_f, 0.0, 1.0)

    # Invert: white (~1.0) -> 0.0, black (~0.0) -> 1.0
    inverted = 1.0 - gray_f  # dark ink becomes large

    # Row intensities: mean over columns -> one value per row
    row_intensities = np.mean(inverted, axis=1)  # shape (H,)

    # Diagnostics: threshold & mean for visualization (tunable constants)
    threshold = 0.1
    mean_intensity = float(np.mean(row_intensities))

    # Make results directory
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"inspect_row_{number}.png")

    # Plot image and profile
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.imshow(gray, cmap='gray', aspect='auto')
    plt.title('Original Image (grayscale)')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.plot(row_intensities, label='Row Intensities', color='blue')
    plt.gca().invert_xaxis() if False else None  # keep natural orientation
    plt.title('Row Intensities (inverted: dark -> high)')
    plt.xlabel('Row index (0 = top)')
    plt.ylabel('Average intensity (0..1)')

    # threshold and mean lines
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.2f}')
    plt.axhline(y=mean_intensity, color='#b58900', linestyle='-', linewidth=2, label=f'Mean = {mean_intensity:.3f}')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    # print saved path for convenience
    print(f"Saved row-intensity visualization to: {out_path}")

    return row_intensities

def find_character_boundaries_by_columns(
        column_intensities: np.ndarray,
        threshold: float = 0.05,
        min_char_width: int = 2,
        min_gap_width: int = 1,
        return_states: bool = False
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
    
    baseline = 0.01
    col = np.asarray(column_intensities).astype(float).ravel()
    n = col.size
    if n == 0:
        return [], 0, (np.array([], dtype=bool) if return_states else None)

    # Normalize to [0,1] if values appear to be in a larger scale (e.g., 0..255)
    if col.max() > 1.0:
        col = col / float(col.max())

    # Per-column boolean: True if column likely contains ink/character
    # intensity_df = pd.DataFrame(column_intensities, columns=['intensity'])
    # states = intensity_df['intensity'] > threshold
    # mean_intensity = np.mean(column_intensities)
   
    
    # mean_intensity = np.mean(column_intensities)
    mean_intensity = np.mean(column_intensities) / 2
    print(f"Mean value is: {mean_intensity}")

    states = col > threshold

    boundaries: List[Tuple[int, int]] = []
    in_character = False
    run_start = 0
    

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



from typing import List, Tuple, Optional
import numpy as np

def find_word_boundaries_by_rows(
    row_intensities: np.ndarray,
    threshold: float = 0.1,
    min_word_height: int = 2,
    min_gap_height: int = 1,
    smoothing: int = 3,
    return_states: bool = False,
) -> Tuple[List[Tuple[int,int]], int, Optional[np.ndarray]]:
    """
    Find word/line boundaries scanning rows (vertical direction).

    Args:
        row_intensities: 1D array-like of per-row intensities (higher => more "ink"/content).
                         Values can be in any scale; function will normalize if max > 1.
        threshold: scalar threshold; a row is considered 'in_word' if intensity > threshold.
        min_word_height: minimum height (rows) for a run to be considered a valid word/line.
        min_gap_height: if the gap between two valid runs is < min_gap_height they will be merged.
        smoothing: odd integer >=1 for simple moving-average smoothing of intensities (helps remove 1-row noise).
        return_states: if True, the function also returns the boolean per-row `in_word` array.

    Returns:
        (boundaries, count, states)
        - boundaries: list of (start_row, end_row) tuples (end is exclusive)
        - count: number of boundaries after merging
        - states: optional 1D boolean array (True = in_word)
    """
    arr = np.asarray(row_intensities).ravel().astype(float)
    n = arr.size
    if n == 0:
        return [], 0, (np.array([], dtype=bool) if return_states else None)

    # Normalize to [0,1] if needed (e.g., values in 0..255)
    if arr.max() < 1.0:
        arr = arr / float(arr.max())
    print("ARR IS: {}".format(arr))
    

    # Simple smoothing (moving average) to reduce single-row spikes
    # if smoothing is None or smoothing <= 1:
    #     smooth = arr
    # else:
    #     k = int(smoothing)
    #     if k % 2 == 0:
    #         k += 1
    #     pad = k // 2
    #     # reflect padding to avoid edge shrinkage
    #     arr_padded = np.pad(arr, pad_width=pad, mode='reflect')
    #     kernel = np.ones(k, dtype=float) / k
    #     smooth = np.convolve(arr_padded, kernel, mode='valid')

    # Compute boolean per-row: True if likely contains content
    threshold =(np.sum(arr)/len(arr))
    print(f"Threshold is: {threshold}")
    states = arr < 0.9
    arr_central = arr[int(len(arr)*0.3):int(len(arr)*0.7)]
    print(f"Len is :{len(arr_central)}")
    threshold_central = (np.sum(arr_central)/len(arr_central))
    print(f"Threshold central: {threshold_central}")

    # states_cental = arr_central < threshold_central
    print("States are: {}".format(states))
    
    # states = 

    # Now scan and collect runs of True values
    boundaries: List[Tuple[int,int]] = []
    in_run = False
    run_start = 0
    for i, v in enumerate(states):
        print("Index: {} Value is: {}".format(i, v))
        if v:
            if not in_run:
                in_run = True
                # if i >= 2:
                    
                run_start = i
                if i > 2:
                   run_start = i-1
                    
                print("start chosen: {}".format(run_start))
            # else: continuing run
        else:
            if in_run:
                run_end = i+1# exclusive
                height = run_end - run_start
                if height >= min_word_height:
                    boundaries.append((run_start, run_end))
                in_run = False
    # finalize trailing run
    if in_run:
        run_end = n
        height = run_end - run_start
        if height >= min_word_height:
            boundaries.append((run_start, run_end))
            
    print("BOUNDARIES BEFORE MERGING: {}".format(boundaries))

    # Merge gaps smaller than min_gap_height
    if len(boundaries) > 1 and min_gap_height > 0:
        merged: List[Tuple[int,int]] = []
        cur_s, cur_e = boundaries[0]
        for s, e in boundaries[1:]:
            gap = s - cur_e
            if gap < min_gap_height:
                # merge
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        boundaries = merged

    count = len(boundaries)
    print("after  function boudaries are: {}".format(boundaries))
    return boundaries, count, (states if return_states else None)

import numpy as np
import cv2
from typing import Optional, Tuple

import numpy as np
import cv2
from typing import Optional, Tuple
from typing import List, Tuple, Optional
import numpy as np

def find_word_boundaries_by_columns(
    column_intensities: np.ndarray,
    threshold: float = 0.05,
    min_word_width: int = 2,
    min_gap_width: int = 1,
    smoothing: int = 3,
    return_states: bool = False,
    verbose: bool = False
) -> Tuple[List[Tuple[int,int]], int, Optional[np.ndarray]]:
    """
    Find word/segment boundaries scanning columns (horizontal direction).

    Args:
        column_intensities: 1D array-like of per-column intensities (higher => more "ink"/content).
                           Values can be any scale; the function will normalize if necessary.
        threshold: scalar threshold; a column is considered 'in_content' if intensity > threshold.
        min_word_width: minimum width (columns) for a run to be considered a valid word/segment.
        min_gap_width: if the gap between two valid runs is < min_gap_width they will be merged.
        smoothing: odd integer >=1 for simple moving-average smoothing (helps remove single-column noise).
        return_states: if True, the function also returns the boolean per-column `in_content` array.
        verbose: if True, prints diagnostic messages (like your previous debug prints).

    Returns:
        (boundaries, count, states)
        - boundaries: list of (start_col, end_col) tuples, end is exclusive (Python-slice friendly).
        - count: number of boundaries after merging.
        - states: optional 1D boolean array (True = in_content).
    """
    arr = np.asarray(column_intensities).ravel().astype(float)
    n = arr.size
    if n == 0:
        return [], 0, (np.array([], dtype=bool) if return_states else None)

    # Normalize to 0..1 if values appear to be larger-scale (e.g., 0..255)
    maxv = arr.max()
    if maxv > 1.0:
        arr = arr / float(maxv)
        if verbose:
            print(f"[columns] Normalized column_intensities by max={maxv:.3f}")

    if verbose:
        print("ARR IS:", arr)

    # Optional smoothing (moving average)
    # if smoothing is None or smoothing <= 1:
    #     smooth = arr
    # else:
    #     k = int(smoothing)
    #     if k % 2 == 0:
    #         k += 1
    #     pad = k // 2
    #     # reflect-padding to avoid boundary shrinkage
    #     arr_padded = np.pad(arr, pad_width=pad, mode='reflect')
    #     kernel = np.ones(k, dtype=float) / k
    #     smooth = np.convolve(arr_padded, kernel, mode='valid')

    # if verbose:
    #     print("SMOOTHED (first 40):", smooth[:min(40, smooth.size)])

    # Determine boolean per-column: True if likely contains content
    states = arr > threshold
    print("ARR: ", arr)
    print("STATES: ", states)
    print("Threhold: ", threshold)
    if verbose:
        print("States are (first 80):", states[:min(80, states.size)])

    # Scan and collect runs of True values
    boundaries: List[Tuple[int,int]] = []
    in_run = False
    run_start = 0
    for i, v in enumerate(states):
        if v:
            if not in_run:
                in_run = True
                run_start = i
                if verbose:
                    print(f"[columns] start chosen: {run_start}")
            # otherwise continue current run
        else:
            if in_run:
                run_end = i  # exclusive
                width = run_end - run_start
                if width >= min_word_width:
                    boundaries.append((run_start, run_end))
                    if verbose:
                        print(f"[columns] appended run ({run_start},{run_end}) width={width}")
                in_run = False
    # finalize trailing run
    if in_run:
        run_end = n
        width = run_end - run_start
        if width >= min_word_width:
            boundaries.append((run_start, run_end))
            if verbose:
                print(f"[columns] appended trailing run ({run_start},{run_end}) width={width}")

    if verbose:
        print("BOUNDARIES BEFORE MERGING:", boundaries)

    # Merge runs separated by tiny gaps (< min_gap_width)
    if len(boundaries) > 1 and min_gap_width > 0:
        merged: List[Tuple[int,int]] = []
        cur_s, cur_e = boundaries[0]
        for s, e in boundaries[1:]:
            gap = s - cur_e
            if gap < min_gap_width:
                # merge this run into current
                cur_e = e
                if verbose:
                    print(f"[columns] merging gap {gap} between ({cur_s},{cur_e})")
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        boundaries = merged

    count = len(boundaries)
    if verbose:
        print("BOUNDARIES AFTER MERGING:", boundaries)
    return boundaries, count, (states if return_states else None)


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert image to single-channel float32 in range [0,1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.issubdtype(img.dtype, np.floating):
        arr = img.astype(np.float32)
        if arr.max() > 1.5:  # likely 0..255 floats
            arr = arr / 255.0
    else:
        arr = img.astype(np.float32) / 255.0
    return arr.clip(0.0, 1.0)


def _find_mask_runs(mask: np.ndarray):
    """Return list of (start, end) runs where mask is True. end is exclusive."""
    runs = []
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            s = i
            i += 1
            while i < n and mask[i]:
                i += 1
            runs.append((s, i))
        else:
            i += 1
    return runs


def crop_to_content(
    img: np.ndarray,
    column_intensities: Optional[np.ndarray] = None,
    center_window: int = 41,
    min_gap_cols: int = 20,
    min_gap_rows: int = 6,
    white_frac_clip: float = 0.15,
    pad: int = 2,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image to content using center-based search for long white runs.

    Args:
      img: input image (H,W) or (H,W,C), dtype uint8 or float; white background assumed.
      column_intensities: optional array length W. If None, computed from image.
      center_window: +/- window size around computed center to establish center intensity.
      min_gap_cols: minimal run length (in pixels) of low-intensity columns to consider as a long white gap.
      min_gap_rows: minimal run length for rows (vertical gaps).
      white_frac_clip: fraction used together with center intensity to define a "white" column threshold.
      pad: pixels to add around detected bbox (clipped to image bounds).
    Returns:
      cropped_img, (top, bottom, left, right)  # bottom/right are exclusive
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")

    h, w = img.shape[:2]
    gray = _to_gray_float(img)  # 1.0 white, 0.0 black

    # column strength: darker -> larger
    if column_intensities is None:
        col_mean = gray.mean(axis=0)   # shape (W,)
        col_strength = 1.0 - col_mean
    else:
        col_strength = np.asarray(column_intensities, dtype=np.float32)
        if col_strength.ndim != 1 or col_strength.size != w:
            raise ValueError("column_intensities must be 1D array of length image width")

    # quick exit: no signal
    if col_strength.sum() == 0:
        return img.copy(), (0, h, 0, w)

    # compute weighted center of mass of column strengths (robust center of content)
    total = col_strength.sum()
    if total <= 0:
        center_idx = w // 2
    else:
        center_idx = int(np.round((np.arange(w) * col_strength).sum() / total))

    # define window around center to estimate typical content strength
    half_w = max(center_window // 2, 1)
    win_l = max(center_idx - half_w, 0)
    win_r = min(center_idx + half_w + 1, w)
    center_mean = float(col_strength[win_l:win_r].mean())
    global_mean = float(col_strength.mean())
    global_max = float(col_strength.max())

    # define white threshold relative to center and global stats
    # white = "small" compared to center content, but allow an absolute floor
    white_threshold = min(center_mean * white_frac_clip + 1e-6, max(global_mean * 0.5, global_max * 0.02))

    # ensure threshold not negative
    white_threshold = max(white_threshold, 1e-6)

    # build boolean mask of white (low-strength) columns
    low_mask_cols = col_strength <= white_threshold

    # find runs of low_mask_cols
    runs_cols = _find_mask_runs(low_mask_cols)

    # find candidate left cut: nearest run to left of center whose length >= min_gap_cols
    left = 0
    right = w
    # filter runs that end <= center_idx (i.e., runs entirely left of center) and length>=min_gap_cols
    left_runs = [(s, e) for s, e in runs_cols if e <= center_idx and (e - s) >= min_gap_cols]
    if left_runs:
        # pick the run with largest e (closest to center)
        best_left_run = max(left_runs, key=lambda re: re[1])
        # content starts at the run end
        left = best_left_run[1]
    else:
        # also consider runs that overlap center (rare): if a run contains center and is long -> we may have no content
        overlapping = [(s, e) for s, e in runs_cols if s <= center_idx < e and (e - s) >= min_gap_cols]
        if overlapping:
            # content region nonexistent => keep full
            return img.copy(), (0, h, 0, w)
        # otherwise, try more permissive: look for runs left of center with slightly shorter length
        left_runs_small = [(s, e) for s, e in runs_cols if e <= center_idx and (e - s) >= max(1, min_gap_cols // 2)]
        if left_runs_small:
            best_left_run = max(left_runs_small, key=lambda re: re[1])
            left = best_left_run[1]

    # find candidate right cut: nearest run to right of center with length>=min_gap_cols
    right_runs = [(s, e) for s, e in runs_cols if s >= center_idx and (e - s) >= min_gap_cols]
    if right_runs:
        best_right_run = min(right_runs, key=lambda re: re[0])  # smallest start => closest to center
        right = best_right_run[0]
    else:
        overlapping = [(s, e) for s, e in runs_cols if s <= center_idx < e and (e - s) >= min_gap_cols]
        if overlapping:
            return img.copy(), (0, h, 0, w)
        right_runs_small = [(s, e) for s, e in runs_cols if s >= center_idx and (e - s) >= max(1, min_gap_cols // 2)]
        if right_runs_small:
            best_right_run = min(right_runs_small, key=lambda re: re[0])
            right = best_right_run[0]

    # ensure we actually found a meaningful interior region
    if right - left <= 1:
        return img.copy(), (0, h, 0, w)

    # Now do **similar approach for rows** but with smaller min_gap_rows
    row_strength = 1.0 - gray.mean(axis=1)  # shape (H,)
    # weighted center row (content center in vertical)
    total_r = row_strength.sum()
    if total_r <= 0:
        center_row = h // 2
    else:
        center_row = int(np.round((np.arange(h) * row_strength).sum() / total_r))

    half_h = max(center_window // 2, 1)
    rwin_l = max(center_row - half_h, 0)
    rwin_r = min(center_row + half_h + 1, h)
    center_row_mean = float(row_strength[rwin_l:rwin_r].mean())
    global_row_mean = float(row_strength.mean())
    global_row_max = float(row_strength.max())

    row_white_thr = min(center_row_mean * white_frac_clip + 1e-6, max(global_row_mean * 0.5, global_row_max * 0.02))
    row_white_thr = max(row_white_thr, 1e-6)
    low_mask_rows = row_strength <= row_white_thr
    runs_rows = _find_mask_runs(low_mask_rows)

    top = 0
    bottom = h
    top_runs = [(s, e) for s, e in runs_rows if e <= center_row and (e - s) >= min_gap_rows]
    if top_runs:
        best_top_run = max(top_runs, key=lambda re: re[1])
        top = best_top_run[1]
    else:
        top_runs_small = [(s, e) for s, e in runs_rows if e <= center_row and (e - s) >= max(1, min_gap_rows // 2)]
        if top_runs_small:
            best_top_run = max(top_runs_small, key=lambda re: re[1])
            top = best_top_run[1]

    bottom_runs = [(s, e) for s, e in runs_rows if s >= center_row and (e - s) >= min_gap_rows]
    if bottom_runs:
        best_bottom_run = min(bottom_runs, key=lambda re: re[0])
        bottom = best_bottom_run[0]
    else:
        bottom_runs_small = [(s, e) for s, e in runs_rows if s >= center_row and (e - s) >= max(1, min_gap_rows // 2)]
        if bottom_runs_small:
            best_bottom_run = min(bottom_runs_small, key=lambda re: re[0])
            bottom = best_bottom_run[0]

    # Expand by pad and clamp
    left = max(left - pad, 0)
    right = min(right + pad, w)
    top = max(top - pad, 0)
    bottom = min(bottom + pad, h)

    # final safety checks
    if right - left <= 1 or bottom - top <= 1:
        return img.copy(), (0, h, 0, w)

    cropped = img[top:bottom, left:right].copy()
    return cropped, (top, bottom, left, right)



def detect_grid_lines(thresh_img):
    # обнаружим линии морфологией
    img_bin = thresh_img.copy()
    # горизонтальные
    kernel_len_h = max(10, img_bin.shape[1]//40)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_h,1))
    horiz = cv2.erode(img_bin, hor_kernel, iterations=1)
    horiz = cv2.dilate(horiz, hor_kernel, iterations=2)

    # вертикальные
    kernel_len_v = max(10, img_bin.shape[0]//40)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,kernel_len_v))
    vert = cv2.erode(img_bin, ver_kernel, iterations=1)
    vert = cv2.dilate(vert, ver_kernel, iterations=2)

    # можно объединить через bitwise_or если надо
    return horiz, vert



import numpy as np
import cv2
from typing import Tuple, List, Optional

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



def make_blobs(img, base_name):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img, removed_cols, removed_rows  = remove_almost_black_borders(img)
 
    horiz, vert = detect_grid_lines(img)
    horiz_mask = cv2.dilate(horiz, np.ones((3,3), np.uint8), iterations=1)
    vert_mask = cv2.dilate(vert, np.ones((3,3), np.uint8), iterations=1)
    
    cv2.imwrite("results/horiz_mask.png", cv2.bitwise_or(horiz_mask, vert_mask))
    # img=preprocess(img,do_blur=Fals
    # img = remove_cell_boundaries(img)
    old_blobs = extract_blobs(img)
    print(f"Len of blobs is: {len(old_blobs)}")
    rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    

    print(len(old_blobs))
    new_word_images = []
    count = -1
    row_intensities = calculate_column_intensity(rot, count, row_calc=True)
    print("Row intensities: \n\n\n\n")
    print(row_intensities)
    print("FINISH")
    # convert back from invert
    w, h = rot.shape
    
    # central_row_intensities = calculate_column_intensity(rot[:, int(h*0.3):h-int(h*0.3)])
    
    row_intensities = 1.0 - row_intensities
    print("Row intensities: \n\n\n\n")
    print(row_intensities)
    boundaries, cnt, states = find_word_boundaries_by_rows(
    row_intensities=row_intensities,
    threshold=0.1,
    min_word_height=2,
    min_gap_height=1,
    smoothing=5,
    return_states=True,
    )
    print("LEN OF BOUNDARIES: ", len(boundaries))
    blobs = []
    count = 0
    for i, (start_col, end_col) in enumerate(boundaries):
        print("Start col: ", start_col, "End col: ", end_col)
        print("Image shape: ", img.shape)
        
        
        if start_col > len(removed_rows):
            start_col -= len(removed_rows)
        if  img.shape[0]- end_col >=2:
            end_col += 2
        elif img.shape[0]- end_col >=1:
            end_col +=1
        if end_col - start_col < 5:
            continue
        if end_col - start_col <=5 and img.shape[0]- end_col >=2:
            end_col += 2
            
        print("New start col: ", start_col, "New end col: ", end_col)
        char_img = img[start_col:end_col, ]

       
        blobs.append(char_img)
        new_word_images.append(char_img)
        

    for blob in blobs:
        count += 1
        # print(blob.shape)
        # if blob is None or blob.size == 0 or blob.shape[0] == 0 or blob.shape[1] == 0:
        #     print(f"⚠️ Skipping empty blob #{count}")
        #     continue  # skip empty blobs
        # column_intensities=calculate_column_intensity(blob, count)
        # cropped, bbox = crop_to_content(blob, column_intensities=column_intensities, pad=1)
        #convert back from inverted
        # cropped = 255 - cropped
        # print()
        cv2.imwrite(f"blobs_2/{base_name}_blob_{count}.png", blob)
        # column_intensities = calculate_column_intensity(blob, count)
        # char_boundaries, count, states = find_character_boundaries_by_columns(
        #     column_intensities,z min_char_width=2, min_gap_width=1, return_states=True
        # )
        # print("LEN OF CHAR BOUNDARIES: ", len(char_boundaries))
        # for i, (start_col, end_col) in enumerate(char_boundaries):
        #     char_img = blob[:, start_col:end_col]
        #     # new_word_images.append(char_img)
        #     cv2.imwrite(f"results/char_{count}_{i}.png", char_img)
        
        print(f"Found {count} characters in blob {count}")
        
        
    count = 0
    if len(blobs) == 0:
        print("NO BLOBS FOUND")
        for blob in old_blobs:
            if blob.shape[0] < 5 or blob.shape[1] < 5:
                continue
            count += 1
            cv2.imwrite(f"blobs/{base_name}_blob_{count}.png", blob)
    print("REMOVED LENGTH: ", len(removed_cols))
    print("REMOVED ROWS: ", len(removed_rows))
    
def make_words(img, base_name):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img, removed_cols, removed_rows  = remove_almost_black_borders(img)
    # rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # row_intensities = 1.0 - row_intensities
    # print("Row intensities: \n\n\n\n")
    # print(row_intensities)
    column_intensities = calculate_column_intensity(img, 1, row_calc=False)
    # column_intensities = 1.0 - column_intensities
    boundaries, cnt, states = find_word_boundaries_by_columns(
    column_intensities=column_intensities,
    threshold=0.01,
    min_word_width=3,
    min_gap_width=1,
    smoothing=5,
    return_states=True
    )
    print("LEN OF BOUNDARIES: ", len(boundaries))
    blobs = []
    
    for i, (start_col, end_col) in enumerate(boundaries):
        print("Start col: ", start_col, "End col: ", end_col)
        print("Image shape: ", img.shape)
        
        if start_col > len(removed_rows):
            start_col -= len(removed_rows)
        if  img.shape[0]- end_col >=2:
            end_col += 2
        elif img.shape[0]- end_col >=1:
            end_col +=1
        if end_col - start_col <=7 and img.shape[0]- end_col >=2:
            end_col += 2
            
        print("New start col: ", start_col, "New end col: ", end_col)
        char_img = img[:,start_col:end_col]
        print("Char img shape: ", char_img.shape)

        if end_col - start_col < 3:
            continue
        blobs.append(char_img)
    count = 0
    for blob in blobs:
        count += 1
        cv2.imwrite(f"words/{base_name}_word_{count}.png", blob)
    
    
    
def remove_black_borders(img: np.ndarray) -> np.ndarray:
    print(img[0,:])
    

        
            
    

if __name__ == "__main__":
    print("I am ")
    working_directory = "blobs_2"
    count = 0
    files_explored = []
    for filename in os.listdir(working_directory):
        # if count > 10:
        #     break
        if filename.endswith('.png'):
            cell_path = os.path.join(working_directory, filename)
            base_name = os.path.splitext(os.path.basename(cell_path))[0]

            img = cv2.imread(cell_path)
            print("Processing: ", cell_path)
            make_words(img, base_name)
            files_explored.append(filename)
        # count += 1
    
    # print("Files explored: ", files_explored)
    
    
    # img = cv2.imread("cells_cleaned/cell_r20_c18.png")
    # make_blobs(img, "cell_r0_c1_blob_4")
    
    
    # img = cv2.imread("cells_production/cell_r20_c15.png")
    
   
        

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
import csv
from test_lines import detect_and_draw_table_lines, group_centroids_by_columns

# ---------- Параметры ----------
INPUT_PATH = "data/input/original.jpeg"   # <-- замените на путь к вашему изображению
OUTPUT_DIR = "cells_production"
DEVELOPMENT_DIR = "steps_out"
os.makedirs(OUTPUT_DIR, exist_ok=True) # to not to conflict

# Порог для объединения линий/точек (в пикселях)
LINE_MERGE_THRESH = 10
POINT_ROW_GROUP_THRESH = 15
POINT_COL_GROUP_THRESH = 15

# ---------- Утилиты ----------
def resize_max(img, max_dim=1600):
    h,w = img.shape[:2]
    scale = 1.0
    if max(h,w) > max_dim:
        scale = max_dim / float(max(h,w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def rotate_90(image):
    """Rotate image 90 degrees clockwise."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


import cv2
import numpy as np
from typing import Tuple

# def upscale_to_size(img_uint8: np.ndarray,
#                     target_size: Tuple[int,int]=(160, 40),
#                     keep_aspect: bool = True,
#                     pad_value: int = 255) -> np.ndarray:
#     """
#     Resize (upscale/downscale) a grayscale image to target_size (width, height).

#     Parameters
#     ----------
#     img_uint8 : np.ndarray
#         Input image (grayscale or BGR). dtype uint8.
#     target_size : tuple (width, height)
#         Target output size in pixels. Default (160, 40).
#     keep_aspect : bool
#         If True, preserve aspect ratio and pad with pad_value to reach target size.
#         If False, stretch image to exactly target_size.
#     pad_value : int
#         Padding pixel value (0..255). Default 255 (white background).

#     Returns
#     -------
#     out : np.ndarray
#         Resized uint8 image of shape (target_height, target_width).
#     """
#     # --- Ensure grayscale ---
#     img = img_uint8
#     if img is None:
#         raise ValueError("img_uint8 is None")
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     target_w, target_h = target_size
#     h, w = img.shape[:2]

#     if not keep_aspect:
#         # Stretch to exact size (fast, may distort)
#         interp = cv2.INTER_CUBIC if (target_w > w or target_h > h) else cv2.INTER_AREA
#         out = cv2.resize(img, (target_w, target_h), interpolation=interp)
#         return out.astype(np.uint8)

#     # Preserve aspect ratio: scale to fit inside target, then pad
#     scale = min(target_w / w, target_h / h)
#     new_w = max(1, int(round(w * scale)))
#     new_h = max(1, int(round(h * scale)))

#     interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
#     resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

#     # create canvas and paste centered
#     canvas = np.full((target_h, target_w), pad_value, dtype=np.uint8)
#     x_off = (target_w - new_w) // 2
#     y_off = (target_h - new_h) // 2
#     canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

#     return canvas

# ---------- Основные шаги ----------
def preprocess(img):
    
    """
    use max_pixels function to remove very light pixels
    then convert to gray (1 channel)
    then apply CLAHE to improve contrast and apply to gray back
    then apply max_pixels function to remove very light pixels
    then apply denoising to remove specks/pixel noise
    then apply thresholding to convert to binary image
    
    """
    
    
    filename_1 = "grayscale_image.png"
    filename_2 = "thresholded_image.png"
    filename_3 = "denoised_image.png"
    path_1 = os.path.join(DEVELOPMENT_DIR, filename_1)
    path_2 = os.path.join(DEVELOPMENT_DIR, filename_2)
    path_3 = os.path.join(DEVELOPMENT_DIR, filename_3)
    
    img = rotate_90(img)

    gray = mask_pixels(img, "mask_1.png")
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
    gray = clahe.apply(gray)
    gray_2 = mask_pixels(img, "mask_2.png")
    den = cv2.bilateralFilter(gray_2, d=9, sigmaColor=75, sigmaSpace=75)
    
    
    # Ensure single-channel uint8 before thresholding
    if len(den.shape) == 3:
        den = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)

    if den.dtype != np.uint8:
        den = (den * 255).astype(np.uint8)
    # # adaptive threshold - хорош для разного освещения
    th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
   
    cv2.imwrite(path_1, gray_2)
    # cv2.imwrite(path_2, th)
    # cv2.imwrite(path_3, den)
    return gray_2, th

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
    
    
    # cv2.imwrite(f"steps_out/{output_path}", result)
    return result

def four_point_transform(img, pts, scale=1.0, auto_rotate=False, rotate_90=False):
    # Order the points consistently: tl, tr, br, bl
    rect = order_quad(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB) * scale)
    maxHeight = int(max(heightA, heightB) * scale)

    # Destination coordinates
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        img,
        M,
        (maxWidth, maxHeight),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Optional: Auto-rotate (simple heuristic using text orientation)
    if auto_rotate:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]

        # Correct the angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate to correct skew
        (h, w) = warped.shape[:2]
        center = (w // 2, h // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        warped = cv2.warpAffine(warped, M_rot, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)

    elif rotate_90:
        # Simple 90-degree rotation (clockwise)
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def find_table_contour(thresh):
    # ищем крупный контур (таблицу)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True) #why to sort? - Why? The table or document is usually the largest object in the image.
    for c in contours:
        peri = cv2.arcLength(c, True) # True perimeter is closed
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            
            plt = approx.reshape(4,2).astype(np.float32)
            
            arr = plt.copy()
            for col in range(arr.shape[1]):
                for i in range(1, arr.shape[0]):
                    if abs(arr[i, col] - arr[i-1, col]) < 20:
                        arr[i, col] = arr[i-1, col] + 3
                        
            return arr
        
    return None


def straighten_table_contour(pts):
    """
    pts: np.array of shape (4,2), the detected contour corners (float32)
    returns: rectified 4 points (float32)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    # take averages for straightening
    top_y = int((rect[0][1] + rect[1][1]) / 2)
    bottom_y = int((rect[2][1] + rect[3][1]) / 2)
    left_x = int((rect[0][0] + rect[3][0]) / 2)
    right_x = int((rect[1][0] + rect[2][0]) / 2)

    # build idealized rectangle
    straight_rect = np.array([
        [left_x, top_y],    # TL
        [right_x, top_y],   # TR
        [right_x, bottom_y],# BR
        [left_x, bottom_y]  # BL
    ], dtype=np.float32)

    return straight_rect

# def warp_table(image, pts):
#     rect = straighten_table_contour(pts)
    
#     width = int(rect[1][0] - rect[0][0])
#     height = int(rect[3][1] - rect[0][1])
    
#     dst = np.array([
#         [0, 0],
#         [width-1, 0],
#         [width-1, height-1],
#         [0, height-1]
#     ], dtype="float32")

#     M = cv2.getPerspectiveTransform(pts, dst)
#     warped = cv2.warpPerspective(image, M, (width, height))
#     return warped, M



def order_quad(pts):
    # упорядочить точки квадрила (tl,tr,br,bl)
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

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


def fill_line_gaps(horiz_mask, vert_mask, img_shape,
                   gap_ratio_h=1/40.0, gap_ratio_v=1/40.0,
                   extra_dilate_after=False):
    """
    Bridge small breaks in horiz_mask and vert_mask.

    - gap_ratio_h: fraction of image width; gaps smaller than width*gap_ratio_h (px) are closed for horizontals
    - gap_ratio_v: fraction of image height; gaps smaller than height*gap_ratio_v (px) are closed for verticals
    - extra_dilate_after: optionally dilate after closing to make lines thicker/continuous

    Returns: horiz_filled, vert_filled
    """
    H, W = img_shape[:2]

    # compute kernel sizes in pixels (must be >=1)
    gap_h_px = max(1, int(round(W * gap_ratio_h)))
    gap_v_px = max(1, int(round(H * gap_ratio_v)))

    # Closing kernel: will bridge gaps up to kernel length
    hor_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_h_px, 1))
    vert_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_v_px))

    horiz_filled = cv2.morphologyEx(horiz_mask, cv2.MORPH_CLOSE, hor_close_kernel, iterations=1)
    vert_filled = cv2.morphologyEx(vert_mask, cv2.MORPH_CLOSE, vert_close_kernel, iterations=1)

    if extra_dilate_after:
        # optional: make lines a bit thicker to help Hough/findContours
        d_h = max(1, gap_h_px // 40)
        d_v = max(1, gap_v_px // 40)
        if d_h > 0:
            horiz_filled = cv2.dilate(horiz_filled, cv2.getStructuringElement(cv2.MORPH_RECT, (1 + d_h, 1)), iterations=1)
        if d_v > 0:
            vert_filled = cv2.dilate(vert_filled, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1 + d_v)), iterations=1)

    return horiz_filled, vert_filled





def visualize_grid_and_cells(rows_of_points, image, out_path):
    """
    Draws nodes, numbers and cell rectangles on a copy of image and saves to out_path
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    rows = rows_of_points
    R = len(rows) - 1
    C = len(rows[0]) - 1

    # draw nodes and indices
    for r_idx, row in enumerate(rows):
        for c_idx, (x,y) in enumerate(row):
            xi, yi = int(round(x)), int(round(y))
            cv2.circle(vis, (xi, yi), 4, (0,0,255), -1)
            cv2.putText(vis, f"{r_idx},{c_idx}", (xi+4, yi-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    # draw cell rectangles
    for r in range(R):
        for c in range(C):
            tl = rows[r][c]; br = rows[r+1][c+1]
            x1,y1 = int(round(tl[0])), int(round(tl[1]))
            x2,y2 = int(round(br[0])), int(round(br[1]))
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 1)

    cv2.imwrite(out_path, vis)
    return out_path


def group_and_visualize_nodes(
        centroids: List[Tuple[float, float]],
        original_img: np.ndarray,
        row_tol: Optional[int] = None,
        trim_to_min_cols: bool = True,
        out_path: str = "steps_out/nodes_visual.png"
    ) -> Tuple[List[List[Tuple[float, float]]], str, Dict]:
    

    if len(centroids) == 0:
        raise ValueError("No centroids provided.")

    H = original_img.shape[0]

    # Default row tolerance
    if row_tol is None:
        row_tol = max(10, H // 150)
        
    row_tol = 8

    diagnostics = {
        "row_tol": int(row_tol),
        "input_centroids_count": len(centroids)
    }

    # Sort points by y, then x
    points = sorted(centroids, key=lambda p: (p[1], p[0]))  # (x, y) → sort by y then x

    # Group into rows by y tolerance
    rows = []
    current_row = [points[0]]
    for (x, y) in points[1:]:
        if abs(y - current_row[-1][1]) <= row_tol:
            current_row.append((x, y))
        else:
            rows.append(current_row)
            current_row = [(x, y)]
    rows.append(current_row)
    # compare size of rows and points
    print(f"Number of rows: {len(rows)}")
    print(f"Number of points: {len(points)}")

    # Sort each row by x
    rows_of_points = [sorted(r, key=lambda pt: pt[0]) for r in rows]


    # --- Visualization ---
    # Prepare overlay (convert grayscale to BGR if needed)
    if len(original_img.shape) == 2:
        overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original_img.copy()

    # Make a slightly dimmed copy and draw on copy (so lines show)
    overlay_draw = overlay.copy()

    # Palette for row colors (BGR)
    palette = [
        (0, 0, 255),    # red
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (128, 0, 255),  # violet-ish
        (0, 128, 255),  # orange-ish
    ]

    # Draw nodes and labels
    for r_idx, row in enumerate(rows_of_points):
        color = palette[r_idx % len(palette)]
        for c_idx, (x, y) in enumerate(row):
            xi, yi = int(round(x)), int(round(y))
            # draw circle
            cv2.circle(overlay_draw, (xi, yi), radius=6, color=color, thickness=-1)
            # draw small white border
            cv2.circle(overlay_draw, (xi, yi), radius=8, color=(255,255,255), thickness=1)
            # label with row,col
            cv2.putText(overlay_draw, f"{r_idx},{c_idx}", (xi+6, yi-6),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                        color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

    # Draw row polylines (connect nodes left->right)
    for r_idx, row in enumerate(rows_of_points):
        pts = np.array([[int(round(x)), int(round(y))] for (x,y) in row], dtype=np.int32)
        if pts.shape[0] >= 2:
            cv2.polylines(overlay_draw, [pts], isClosed=False, color=(250,0,0), thickness=1, lineType=cv2.LINE_AA)

    # Draw column polylines (connect nodes top->bottom) using trimmed min_len
    if len(rows_of_points) >= 2 and len(rows_of_points[0]) >= 1:
        ncols = len(rows_of_points[0])
        for c in range(ncols):
            col_pts = []
            for r in range(len(rows_of_points)):
                col_pts.append((int(round(rows_of_points[r][c][0])), int(round(rows_of_points[r][c][1]))))
            col_np = np.array(col_pts, dtype=np.int32)
            if col_np.shape[0] >= 2:
                cv2.polylines(overlay_draw, [col_np], isClosed=False, color=(250,0,0), thickness=1, lineType=cv2.LINE_AA)

    # Save visualization
    cv2.imwrite(out_path, overlay_draw)

    return rows_of_points, out_path, diagnostics


def group_centroids_into_grid(centroids, row_tol=None, min_cols=3, debug=False):
    """
    Groups centroids (x,y) into grid rows and columns.

    Args:
        centroids: list of (x, y) points
        row_tol: tolerance in pixels for grouping by Y (auto-computed if None)
        min_cols: minimum number of columns to accept as valid row
        debug: if True, prints intermediate info

    Returns:
        rows_pts: list of rows (each row = list of (x, y) tuples)
    """
    if len(centroids) == 0:
        raise ValueError("No centroids provided.")

    # --- Sort by Y (vertical position)
    points = sorted(centroids, key=lambda p: (p[1], p[0]))
    H = max(p[1] for p in points) - min(p[1] for p in points)
    if row_tol is None:
        row_tol = max(8, int(H // 120))  # adaptive tolerance

    rows = []
    current_row = [points[0]]
    for (x, y) in points[1:]:
        # same row if y difference is small
        if abs(y - current_row[-1][1]) <= row_tol:
            current_row.append((x, y))
        else:
            if len(current_row) >= min_cols:  # skip tiny noise rows
                rows.append(current_row)
            current_row = [(x, y)]
    if len(current_row) >= min_cols:
        rows.append(current_row)

    # --- Sort each row left → right
    rows = [sorted(r, key=lambda p: p[0]) for r in rows]

    # # --- Normalize rows to equal length (trim to min)
    min_len = min(len(r) for r in rows)
    # rows = [r[:min_len] for r in rows if len(r) >= min_len]

   

    return rows



def visualize_rows(image, rows_pts, output_path=None):
    """
    Visualize detected rows by drawing horizontal lines at the mean y of each row.
    
    Parameters:
        image: Input image (grayscale or BGR). If grayscale, it will be converted to BGR for color drawing.
        rows_pts: List of rows; each row is a list of [x, y] points.
        output_path: Optional path to save the image.
    """
    # Ensure image is 3-channel BGR for colored drawing
    if len(image.shape) == 2:
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = image.copy()

    # Draw each row
    for i, row in enumerate(rows_pts):
        if not row:
            continue
        # Compute average y for the row
        y_coords = [pt[1] for pt in row]
        avg_y = int(np.mean(y_coords))
        
        # Choose a color (cycle through a few)
        color = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ][i % 5]

        # Draw horizontal line across the image width
        cv2.line(vis_img, (0, avg_y), (vis_img.shape[1] - 1, avg_y), color, thickness=2)

        # Optionally: draw each centroid in the row
        for (x, y) in row:
            cv2.circle(vis_img, (int(x), int(y)), radius=4, color=color, thickness=-1)

        # Optionally: label the row
        cv2.putText(vis_img, f"Row {i}", (10, avg_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"Saved row visualization to: {output_path}")
    
    return vis_img



import os
import cv2
import numpy as np


# === USAGE ===

def _to_int_point(p):
    if p is None:
        return None
    return (int(round(float(p[0]))), int(round(float(p[1]))))

def find_nearest_index_by_x(row, x, x_tol):
    """Find index i in row where abs(row[i].x - x) <= x_tol and return i. 
       If multiple matches, return the one with minimal abs(x - row[i].x).
       row: list of (x,y) or None. Returns None if no match."""
    best_i = None
    best_dx = None
    for i, pt in enumerate(row):
        if pt is None:
            continue
        dx = abs(pt[0] - x)
        if dx <= x_tol and (best_dx is None or dx < best_dx):
            best_dx = dx
            best_i = i
    return best_i

import os
import cv2
import numpy as np

def extract_table_cells_with_merge_detection(working_img, centroids_np, rows_pts, 
                                              output_dir="cells_production", padding=0):
    """
    Extract cells from a table, detecting and handling merged cells that span multiple rows/columns.
    
    Parameters:
    - working_img: The table image
    - centroids_np: Array of all intersection points
    - rows_pts: Grouped centroids organized by rows
    - output_dir: Directory to save cell images
    - padding: Pixels to add inside cell boundaries to avoid lines
    
    Returns:
    - cells_list: List of dicts with cell info including position, span, and image
    - grid_map: 2D array showing which cells occupy which grid positions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    x_tol = 6
    
    # Convert to grayscale for line detection
    if len(working_img.shape) == 3:
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = working_img.copy()
    
    # Group centroids into columns
    cols_pts = group_centroids_by_columns(centroids_np, col_tol=4)
    h, w = (working_img.shape[0], working_img.shape[1]) if hasattr(working_img, 'shape') else (0, 0)

    
    print(f"\n=== Extracting Table Cells with Merge Detection ===")
    print(f"Grid size: {len(rows_pts)} rows × {len(cols_pts)} columns")
    
    # Sort rows and columns
    rows_sorted = sorted(rows_pts, key=lambda row: np.mean([pt[1] for pt in row]))
    cols_sorted = sorted(cols_pts, key=lambda col: np.mean([pt[0] for pt in col]))
    
    from try_extract import visualize_code
    visualize_code(rows_sorted, cols_sorted)
    
        
    norm_rows = [[_to_int_point(p) for p in row] for row in rows_sorted]
    processed = set()
    cells = []
    
    debug_img = None
    
    # Make a color copy for debug drawing (if grayscale, convert to BGR)
    if len(working_img.shape) == 2:
        debug_img = cv2.cvtColor(working_img, cv2.COLOR_GRAY2BGR)
    else:
        debug_img = working_img.copy()
    
    cells_list = []
    count = 0
     # Loop rows (except last) and columns (except last)
    for r_idx in range(len(norm_rows) - 1):
        row_top = norm_rows[r_idx]
        row_bottom = norm_rows[r_idx + 1]
        
        # number of usable columns for this row pair is min(lengths) - 1
        usable_cols = min(len(row_top), len(row_bottom)) - 1
        if usable_cols <= 0:
            continue
        
        
        for c_idx in range(usable_cols):
            
            row_top = norm_rows[r_idx]
            # iterate c_idx across top row gaps
            for c_idx in range(len(row_top) - 1):
                if (r_idx, c_idx) in processed:
                    continue

                tl = row_top[c_idx]
                tr = row_top[c_idx + 1]
                if tl is None or tr is None:
                    continue

                # search downward for a row r2 that contains matches for both tl.x and tr.x
                found_bottom = None
                found_bl_i = None
                found_br_i = None
                for r2 in range(r_idx + 1, len(norm_rows)):
                    cand_row = norm_rows[r2]
                    # find nearest indices by x in candidate row
                    bl_i = find_nearest_index_by_x(cand_row, tl[0], x_tol)
                    br_i = find_nearest_index_by_x(cand_row, tr[0], x_tol)
                    if bl_i is not None and br_i is not None:
                        # ensure bottom is below top (y should be larger)
                        if cand_row[bl_i][1] > tl[1] and cand_row[br_i][1] > tl[1]:
                            found_bottom = r2
                            found_bl_i = bl_i
                            found_br_i = br_i
                            break

                if found_bottom is None:
                    # no suitable bottom found -- fallback: use the immediate next row if available
                    fallback_row = norm_rows[r_idx + 1]
                    bl_i = find_nearest_index_by_x(fallback_row, tl[0], x_tol)
                    br_i = find_nearest_index_by_x(fallback_row, tr[0], x_tol)
                    if bl_i is None or br_i is None:
                        # can't form a reliable cell, skip
                        continue
                    found_bottom = r_idx + 1
                    found_bl_i = bl_i
                    found_br_i = br_i

                # determine bottom points
                row_bottom = norm_rows[found_bottom]
                bl = row_bottom[found_bl_i]
                br = row_bottom[found_br_i]
                if bl is None or br is None:
                    continue

                # Determine column span: bottom indices may not equal top indices
                # If bottom indices are not in order (br < bl), swap
                if found_br_i < found_bl_i:
                    found_bl_i, found_br_i = found_br_i, found_bl_i
                    bl, br = br, bl
                    
                # Compute bounding rectangle from four corners
                xs = [tl[0], tr[0], br[0], bl[0]]
                ys = [tl[1], tr[1], br[1], bl[1]]
                x_min = int(max(0, min(xs) - padding))
                y_min = int(max(0, min(ys) - padding))
                x_max = int(min(w, max(xs) + padding))
                y_max = int(min(h, max(ys) + padding))
                
                if count <2:
                    count += 1

                # Validate size
                if x_max <= x_min or y_max <= y_min:
                    # degenerate box, skip
                    continue
                
                # Crop (preserve original channels)
                cell_img = working_img[y_min:y_max, x_min:x_max].copy()

                if cell_img.size == 0:
                    continue
                
                # Save file
                filename = f"cell_r{r_idx}_c{c_idx}.png"
                filepath = os.path.join(output_dir, filename)
                # Use PNG to avoid compression artifacts
                cv2.imwrite(filepath, cell_img)
                
                # Collect metadata
                meta = {
                    "filename": filename,
                    "r": int(r_idx),
                    "c": int(c_idx),
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                    "width": int(x_max - x_min),
                    "height": int(y_max - y_min)
                }
                cells_list.append(meta)
                
                # Debug draw
                
                # draw rectangle
                cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                # label
                label = f"{r_idx},{c_idx}"
                # put label at top-left inside box (clamp)
                tx, ty = x_min + 3, min(y_min + 12, h-1)
                cv2.putText(debug_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
                # else:
                #     continue
    
    # Save manifest CSV
    if len(cells_list) > 0:
        manifest_path = os.path.join(output_dir, "manifest.csv")
        keys = ["filename","r","c","x_min","y_min","x_max","y_max","width","height"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in cells_list:
                writer.writerow(row)
    
    # num_rows = len(rows_pts) - 1
    # num_cols = len(rows_pts[0]) - 1
        

    
    return cells_list, None
    

def cells_to_csv_structure(cells_list, grid_map):
    """
    Convert extracted cells into a CSV-compatible structure.
    Handles merged cells appropriately.
    
    Returns:
    - 2D list representing the CSV structure with cell references
    """
    if not grid_map:
        return []
    
    num_rows = len(grid_map)
    num_cols = len(grid_map[0]) if grid_map else 0
    
    # Create CSV structure
    csv_structure = []
    
    for r in range(num_rows):
        row_data = []
        for c in range(num_cols):
            cell_id = grid_map[r][c]
            
            if cell_id == -1:
                row_data.append({"cell_id": None, "content": "", "is_merged_continuation": False})
            else:
                # Find the cell data
                cell = next((cell for cell in cells_list if cell["id"] == cell_id), None)
                
                if cell:
                    # Check if this is the top-left position of the cell
                    is_origin = (r == cell["row"] and c == cell["col"])
                    
                    row_data.append({
                        "cell_id": cell_id,
                        "content": f"[Cell {cell_id}]",  # Placeholder for OCR text
                        "is_merged_continuation": not is_origin,
                        "origin_row": cell["row"],
                        "origin_col": cell["col"],
                        "row_span": cell["row_span"],
                        "col_span": cell["col_span"],
                        "image_path": cell["path"]
                    })
                else:
                    row_data.append({"cell_id": None, "content": "", "is_merged_continuation": False})
        
        csv_structure.append(row_data)
    
    return csv_structure



# ---------- Main pipeline ----------
def process_table_image(path):
    # read image  and resize 
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img, scale = resize_max(img, max_dim=1600)
    


    gray, th = preprocess(img)

    # пробуем найти контур таблицы и сделать warp
    quad = find_table_contour(th)
    
    cv2.drawContours(gray, [quad.astype(int)], -1, (0, 255, 0), 2)
    cv2.imwrite("steps_out/quad_check.png", gray)
    # lines_img = gray.copy()
    if quad is not None:
        
        # warped_gray, M = warp_table(gray, quad)
        
        warped_gray = four_point_transform(gray, quad)
        
        if len(warped_gray.shape) == 3:
            warped_gray = cv2.cvtColor(warped_gray, cv2.COLOR_BGR2GRAY)

        # Since warped_gray is already grayscale:
        _, th_w = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        thresh_for_lines = th_w
        working_img = warped_gray

        cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "warped.png"), working_img)
        working_img= mask_pixels(warped_gray)
        # cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "warped.png"), working_img)
    else:
        # если не нашли контур — работаем с оригиналом
        thresh_for_lines = th
        working_img =  gray

    horiz, vert = detect_grid_lines(thresh_for_lines)
    
    horiz_mask, vert_mask = fill_line_gaps(horiz, vert, working_img.shape, gap_ratio_h=1/600.0, gap_ratio_v=1/600.0)
    # beofre 20 and 7 
    horiz_mask = cv2.dilate(horiz_mask, np.ones((3,3), np.uint8), iterations=1)
    vert_mask = cv2.dilate(vert_mask, np.ones((3,3), np.uint8), iterations=1)
    
    cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "horiz_mask.png"), cv2.bitwise_or(horiz_mask, vert_mask))

    
    intersections = cv2.bitwise_and(horiz_mask, vert_mask)
    cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "intersections.png"), intersections)
    
   
    # найти центры connectedComponents
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections, connectivity=8)
    # # centroids — список точек (x,y)
    # Make a copy so we don't modify original
    img_with_points = working_img.copy()
    
    centroids_np = np.array(centroids)  # from your earlier step


    # Draw each centroid as a small red circle
    for cx, cy in centroids:
        cv2.circle(img_with_points, (int(cx), int(cy)), radius=3, color=(0, 0, 255), thickness=-1)  # Red dot
        
    cv2.imwrite("steps_out/intersections_visualized.png", img_with_points)

        
    # rows_pts, viz_path, info = group_and_visualize_nodes(centroids, working_img, area_thresh=0, row_tol=None)
    rows_pts, viz_path, info = group_and_visualize_nodes(
        centroids=centroids_np,
        original_img=img_with_points,
        row_tol=8,
        trim_to_min_cols=True,
        out_path="steps_out/intersections_grouped.png"
    )
    print("Saved debug v gisualization to:", viz_path)
    
    print("Detected rows x cols:", len(rows_pts), len(rows_pts[0]) if rows_pts else 0)
    
    
    rows_pts = group_centroids_into_grid(centroids_np, row_tol=4, debug=True)
    visualize_rows(working_img, rows_pts, "steps_out/rows_visualized.png")
    
    # display parameters of rows_pts / characterists 

    
    # After your existing code:
    result_img = detect_and_draw_table_lines(working_img, centroids_np, rows_pts)
    cv2.imwrite("steps_out/lines_detected.png", result_img)
    # Ensure it's grayscale
    gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY) if len(result_img.shape) == 3 else result_img

    # Ensure it's 8-bit unsigned int
    gray = cv2.convertScaleAbs(gray)

    # Apply adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV, 11, 2)

    
    horiz, vert = detect_grid_lines(th)
    
    horiz_mask, vert_mask = fill_line_gaps(horiz, vert, result_img.shape, gap_ratio_h=1/600.0, gap_ratio_v=1/600.0)
    # beofre 20 and 7 
    horiz_mask = cv2.dilate(horiz_mask, np.ones((3,3), np.uint8), iterations=1)
    vert_mask = cv2.dilate(vert_mask, np.ones((3,3), np.uint8), iterations=1)
    
    cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "horiz_mask_after.png"), cv2.bitwise_or(horiz_mask, vert_mask))
    intersections = cv2.bitwise_and(horiz_mask, vert_mask)
    cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "intersections.png"), intersections)
    
   
    # найти центры connectedComponents
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections, connectivity=8)

    img_with_points = result_img.copy()
    
    centroids_np = np.array(centroids)  # from your earlier step


    # Draw each centroid as a small red circle
    for cx, cy in centroids:
        cv2.circle(img_with_points, (int(cx), int(cy)), radius=3, color=(0, 0, 255), thickness=-1)  # Red dot
        
    cv2.imwrite("steps_out/intersections_visualized.png", img_with_points)

        
    # rows_pts, viz_path, info = group_and_visualize_nodes(centroids, working_img, area_thresh=0, row_tol=None)
    rows_pts, viz_path, info = group_and_visualize_nodes(
        centroids=centroids_np,
        original_img=img_with_points,
        row_tol=8,
        trim_to_min_cols=True,
        out_path="steps_out/intersections_grouped.png"
    )
    print("Saved debug v gisualization to:", viz_path)
    print("Detected rows x cols:", len(rows_pts), len(rows_pts[0]) if rows_pts else 0)
    
    
    rows_pts = group_centroids_into_grid(centroids_np, row_tol=4, debug=True)
    print("rows_pts:", rows_pts)
    print("SECOND Detected rows x cols:", len(rows_pts), len(rows_pts[0]) if rows_pts else 0)
    visualize_rows(working_img, rows_pts, "steps_out/rows_visualized.png")
    
    # optional visualize nodes on the warped image:
    viz_path = os.path.join("steps_out", "grid_nodes_and_cells.png")
    os.makedirs("steps_out", exist_ok=True)
    visualize_grid_and_cells(rows_pts, result_img, viz_path)
    print("Saved visualization:", viz_path)
        
    cells_list, grid_map = extract_table_cells_with_merge_detection(
        result_img, centroids_np, rows_pts
    )
    
    csv_structure = cells_to_csv_structure(cells_list, grid_map)



# ---------- Запуск ----------
if __name__ == "__main__":
    process_table_image(INPUT_PATH)

    print("Готово. Ячейки сохранены в:", OUTPUT_DIR)

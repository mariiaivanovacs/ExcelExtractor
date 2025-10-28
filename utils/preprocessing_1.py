# NOT IN FINAL PIPELINE


import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
import csv


# ---------- Параметры ----------
INPUT_PATH = "data/input/original.jpeg"   # <-- замените на путь к вашему изображению
OUTPUT_DIR = "cells_out"
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

def sort_points_by_row_col(points, row_thresh=POINT_ROW_GROUP_THRESH, col_thresh=POINT_COL_GROUP_THRESH):
    """
    points: list of (x,y)
    Возвращает: matrix (list of rows) где каждая строка — список точек слева направо.
    Группирует точки по y (строки), затем сортирует внутри строки по x.
    """
    pts = np.array(points)
    # group by y
    ys = pts[:,1]
    sorted_idx = np.argsort(ys)
    pts_sorted = pts[sorted_idx]

    rows = []
    current_row = [tuple(pts_sorted[0])]
    for p in pts_sorted[1:]:
        if abs(p[1] - current_row[-1][1]) <= row_thresh:
            current_row.append(tuple(p))
        else:
            rows.append(current_row)
            current_row = [tuple(p)]
    rows.append(current_row)

    # sort points within each row by x
    for i, r in enumerate(rows):
        rows[i] = sorted(r, key=lambda x: x[0])

    return rows

def merge_similar_lines(lines, axis=0, thresh=LINE_MERGE_THRESH):
    """
    lines: list of lines in form (x1,y1,x2,y2)
    axis=0 -> merge vertical-ish (x coords), axis=1 -> horizontal-ish (y coords)
    Возвращает усреднённые линии
    """
    if len(lines) == 0:
        return []
    # represent line by its midpoint coordinate along axis (x for vertical, y for horizontal)
    coords = []
    for l in lines:
        x1,y1,x2,y2 = l
        coords.append(((x1+x2)/2.0, l))
    coords.sort(key=lambda x: x[0])
    merged = []
    current_group = [coords[0][1]]
    current_center = coords[0][0]
    for c, l in coords[1:]:
        if abs(c - current_center) <= thresh:
            current_group.append(l)
            current_center = (current_center * (len(current_group)-1) + c) / len(current_group)
        else:
            # average group
            xs1 = [x1 for x1,y1,x2,y2 in current_group]
            ys1 = [y1 for x1,y1,x2,y2 in current_group]
            xs2 = [x2 for x1,y1,x2,y2 in current_group]
            ys2 = [y2 for x1,y1,x2,y2 in current_group]
            merged.append((int(np.mean(xs1)), int(np.mean(ys1)), int(np.mean(xs2)), int(np.mean(ys2))))
            current_group = [l]
            current_center = c
    # last
    xs1 = [x1 for x1,y1,x2,y2 in current_group]
    ys1 = [y1 for x1,y1,x2,y2 in current_group]
    xs2 = [x2 for x1,y1,x2,y2 in current_group]
    ys2 = [y2 for x1,y1,x2,y2 in current_group]
    merged.append((int(np.mean(xs1)), int(np.mean(ys1)), int(np.mean(xs2)), int(np.mean(ys2))))
    return merged

def rotate_90(image):
    """Rotate image 90 degrees clockwise."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

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
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)

            xs = pts[:,0]; ys = pts[:,1]
            left = int(xs.min()); right = int(xs.max())
            top = int(ys.min()); bottom = int(ys.max())
            rect = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
        
            return rect
    return None

# def find_table_contour(thresh, debug=False):
#     """
#     Try to find a 4-point table contour. If the contour approach fails,
#     use a projection fallback that extracts horizontal & vertical lines
#     and finds row/col spans with many "line" pixels.
#     - thresh: binary image (H,W) with foreground as 255 (or 0/1).
#     Returns: 4x2 float32 points in order [tl, tr, br, bl] (or rectangle order [left,top],[right,top],[right,bottom],[left,bottom])
#              or None if nothing found.
#     """
#     # normalize to uint8 binary image (0/255)
#     t = np.array(thresh, copy=True)
#     if t.dtype != np.uint8:
#         t = (t * 255).astype(np.uint8) if t.max() <= 1 else t.astype(np.uint8)
#     _, t_bin = cv2.threshold(t, 127, 255, cv2.THRESH_BINARY)

#     H, W = t_bin.shape[:2]

#     # 1) contour-based detection (original logic)
#     contours_info = cv2.findContours(t_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # OpenCV version differences
#     contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
#     if contours:
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         for c in contours:
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#             if len(approx) == 4:
#                 pts = approx.reshape(4, 2).astype(np.float32)
#                 # return as tl,tr,br,bl if you prefer; here we return rectangle-like ordering:
#                 # convert to [left,top],[right,top],[right,bottom],[left,bottom]
#                 xs = pts[:,0]; ys = pts[:,1]
#                 left = int(xs.min()); right = int(xs.max())
#                 top = int(ys.min()); bottom = int(ys.max())
#                 rect = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
#                 return rect

#     2) Projection fallback using morphological horizontal/vertical extraction
#     heuristic kernel sizes (adjustable)
#     horiz_len = max(10, W // 15)
#     vert_len  = max(10, H // 15)

#     horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
#     vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

#     # Extract horizontal lines
#     horiz = cv2.erode(t_bin, horiz_kernel, iterations=1)
#     horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)

#     # Extract vertical lines
#     vert = cv2.erode(t_bin, vert_kernel, iterations=1)
#     vert = cv2.dilate(vert, vert_kernel, iterations=2)

#     # Use the horizontal mask (horiz) to compute row sums and vertical mask for col sums.
#     row_sum = np.sum(horiz == 255, axis=1)  # counts per row
#     col_sum = np.sum(vert == 255, axis=0)   # counts per col

#     # thresholds as fraction of max
#     row_max = np.max(row_sum) if row_sum.size else 0
#     col_max = np.max(col_sum) if col_sum.size else 0
#     row_thresh = 0.3 * row_max if row_max > 0 else 0
#     col_thresh = 0.3 * col_max if col_max > 0 else 0

#     # find contiguous runs above threshold
#     def find_span(arr, thr, min_run=10):
#         above = np.where(arr > thr)[0]
#         if above.size == 0:
#             return None
#         # group contiguous indices
#         groups = np.split(above, np.where(np.diff(above) != 1)[0] + 1)
#         # select the longest group
#         groups = sorted(groups, key=lambda g: g.size, reverse=True)
#         for g in groups:
#             if g.size >= min_run:
#                 return int(g[0]), int(g[-1])
#         # fallback: return full min/max if any exist
#         return int(above[0]), int(above[-1])

#     row_span = find_span(row_sum, row_thresh, min_run=max(5, H // 80))
#     col_span = find_span(col_sum, col_thresh, min_run=max(5, W // 80))

#     if row_span is not None and col_span is not None:
#         top, bottom = row_span
#         left, right = col_span
#         # clamp to image bounds
#         top = max(0, top); bottom = min(H - 1, bottom)
#         left = max(0, left); right = min(W - 1, right)
#         pts = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
#         return pts

#     # nothing found
#     if debug:
#         print("find_table_contour: no table found (contour and projection fallback failed).")
#     return None

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

# def four_point_transform(img, pts, scale=1.0):
#     # Order the points consistently: tl, tr, br, bl
#     rect = order_quad(pts)
#     (tl, tr, br, bl) = rect

#     # Compute width and height
#     widthA = np.linalg.norm(br - bl)
#     widthB = np.linalg.norm(tr - tl)
#     heightA = np.linalg.norm(tr - br)
#     heightB = np.linalg.norm(tl - bl)

#     maxWidth = int(max(widthA, widthB) * scale)
#     maxHeight = int(max(heightA, heightB) * scale)

#     # Destination coordinates
#     dst = np.array([
#         [0, 0],
#         [maxWidth - 1, 0],
#         [maxWidth - 1, maxHeight - 1],
#         [0, maxHeight - 1]], dtype="float32")

#     # Perspective transform
#     M = cv2.getPerspectiveTransform(rect, dst)
#     warped = cv2.warpPerspective(
#         img,
#         M,
#         (maxWidth, maxHeight),
#         flags=cv2.INTER_NEAREST,
#         borderMode=cv2.BORDER_REPLICATE
#     )

#     return warped


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
def lines_from_mask(mask, min_line_length=50):
    # Детектим линии через Hough на маске (Canny не нужен, использую маску)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                              minLineLength=min_line_length, maxLineGap=20)
    out = []
    if lines_p is None:
        return out
    for line in lines_p:
        x1,y1,x2,y2 = line[0]
        out.append((x1,y1,x2,y2))
    return out

def line_intersection(l1, l2):
    # l = (x1,y1,x2,y2)
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (int(round(px)), int(round(py)))

def extract_cells_from_points(rows_of_points, image, out_dir=OUTPUT_DIR):
    """
    rows_of_points: list of rows; each row is sorted points left->right
    image: warped top-down image
    returns: list of dicts {'row':i,'col':j,'coords':(tl,tr,br,bl),'path':...}
    """
    cells = []
    rows = rows_of_points
    n_rows = len(rows)-1
    n_cols = len(rows[0])-1
    for r in range(n_rows):
        for c in range(n_cols):
            tl = rows[r][c]
            tr = rows[r][c+1]
            bl = rows[r+1][c]
            br = rows[r+1][c+1]
            # perspective transform each cell to rectangle
            src = np.array([tl, tr, br, bl], dtype='float32')
            w = int(np.linalg.norm(tr - tl))
            h = int(np.linalg.norm(bl - tl))
            if w <=0 or h <=0:
                continue
            dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype='float32')
            M = cv2.getPerspectiveTransform(src, dst)
            cell_img = cv2.warpPerspective(image, M, (w,h))
            filename = f"cell_r{r}_c{c}.png"
            path = os.path.join(out_dir, filename)
            cv2.imwrite(path, cell_img)
            cells.append({'row': r, 'col': c, 'coords': (tuple(tl), tuple(tr), tuple(br), tuple(bl)), 'path': path})
    return cells



def build_rows_from_centroids(centroids, img_shape,
                              area_thresh=0,
                              row_tol=None,
                              trim_to_min=True,
                              min_points_per_row=2):
    """
    centroids: array-like of (x,y) floats
    img_shape: working_img.shape
    Returns: rows_of_points (list of rows; each row is list of (x,y) floats)
    """
    H, W = img_shape[:2]
    pts = np.asarray(centroids, dtype='float32')
    if pts.size == 0:
        return []

    # defaults
    if row_tol is None:
        row_tol = max(10, H // 150)   # increase if rows split too many times

    # sort by y then x
    pts_sorted = pts[np.lexsort((pts[:,0], pts[:,1]))]  # sort by y then x

    rows = []
    current = [tuple(pts_sorted[0])]
    for (x, y) in pts_sorted[1:]:
        if abs(y - current[-1][1]) <= row_tol:
            current.append((float(x), float(y)))
        else:
            # finalize row (sort by x)
            current = sorted(current, key=lambda p: p[0])
            if len(current) >= min_points_per_row:
                rows.append(current)
            current = [(float(x), float(y))]
    # last row
    current = sorted(current, key=lambda p: p[0])
    if len(current) >= min_points_per_row:
        rows.append(current)

    if len(rows) < 2:
        # not enough rows to form cells
        return rows

    # optionally trim each row to the same number of columns (min length)
    if trim_to_min:
        min_len = min(len(r) for r in rows)
        rows = [r[:min_len] for r in rows]

    return rows

def save_cells_and_csv(rows_of_points, image, out_dir="output/cells"):
    """
    rows_of_points: list of rows (each row sorted left->right), length = R+1, cols = C+1
    image: warped image (grayscale or color)
    Saves cell images and a CSV with coordinates.
    Returns list of cell dicts (like extract_cells_from_points does).
    """
    os.makedirs(out_dir, exist_ok=True)
    # convert to numpy arrays as extract_cells expects
    rows_np = [np.array(r, dtype='float32') for r in rows_of_points]

    # If you have your extract_cells_from_points function, call it:
    try:
        cells = extract_cells_from_points(rows_np, image, out_dir=out_dir)
    except NameError:
        # If extract_cells_from_points is not in scope, do a simple rectangular crop fallback
        cells = []
        n_rows = len(rows_np) - 1
        n_cols = rows_np[0].shape[0] - 1
        for r in range(n_rows):
            for c in range(n_cols):
                tl = rows_np[r][c]; br = rows_np[r+1][c+1]
                x1,y1 = map(int, tl); x2,y2 = map(int, br)
                x1,x2 = sorted([max(0,x1), min(image.shape[1], x2)])
                y1,y2 = sorted([max(0,y1), min(image.shape[0], y2)])
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                path = os.path.join(out_dir, f"cell_r{r}_c{c}.png")
                cell_img = image[y1:y2, x1:x2]
                cv2.imwrite(path, cell_img)
                cells.append({'row': r, 'col': c,
                              'coords': ((x1,y1),(x2,y1),(x2,y2),(x1,y2)),
                              'path': path})

    # write CSV
    csv_path = os.path.join(out_dir, "cells_coords.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row','col','x1','y1','x2','y2'])
        for c in cells:
            r = c['row']; col = c['col']
            tl, tr, br, bl = c['coords']
            x1,y1 = map(int, tl[:2])
            x2,y2 = map(int, br[:2])
            writer.writerow([r, col, x1, y1, x2, y2])

    return cells, csv_path


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
    
    """
    - intersections_mask: binary mask (0/255) containing intersection pixels (result of bitwise_and of horiz/vert)
    - original_img: original image (grayscale or BGR) used for overlay visualization
    - area_thresh: min area (in px) to keep a connected component. If None, computed heuristically.
    - row_tol: tolerance in pixels for grouping centroids into same row (if None computed as max(10, h//150))
    - trim_to_min_cols: if True, trims all rows to the minimum number of columns found
    - out_path: where to save the visualization PNG

    Returns:
        rows_of_points: list of rows, each row is a list of (x,y) tuples (float)
        viz_img_path: path to saved visualization image
        diagnostics: dict with some stats
    """
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
    # compare size of rosw and points
    print(f"Number of rows: {len(rows)}")
    print(f"Number of points: {len(points)}")

    # Sort each row by x
    rows_of_points = [sorted(r, key=lambda pt: pt[0]) for r in rows]

    # # Optionally trim rows to min length
    # if trim_to_min_cols and rows_of_points:
    #     min_len = min(len(r) for r in rows_of_points)
    #     rows_of_points = [r[:min_len] for r in rows_of_points]
    # else:
    #     min_len = min(len(r) for r in rows_of_points) if rows_of_points else 0

    # diagnostics.update({
    #     "rows_detected": len(rows_of_points),
    #     "cols_per_row_used": min_len,
    #     "points_kept": sum(len(r) for r in rows_of_points)
    # })

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




def extract_table_cells_with_merge_detection(working_img, centroids_np, rows_pts, 
                                              output_dir="cells_out", padding=2):
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
    
    # Convert to grayscale for line detection
    if len(working_img.shape) == 3:
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = working_img.copy()
    
    # Group centroids into columns
    cols_pts = group_centroids_by_columns(centroids_np, col_tol=4)
    
    print(f"\n=== Extracting Table Cells with Merge Detection ===")
    print(f"Grid size: {len(rows_pts)} rows × {len(cols_pts)} columns")
    
    # Sort rows and columns
    rows_sorted = sorted(rows_pts, key=lambda row: np.mean([pt[1] for pt in row]))
    cols_sorted = sorted(cols_pts, key=lambda col: np.mean([pt[0] for pt in col]))
    
    # Build intersection grid
    grid = build_intersection_grid(rows_sorted, cols_sorted)
    
    num_rows = len(grid) - 1
    num_cols = len(grid[0]) - 1
    
    # Track which grid cells have been processed
    processed = [[False for _ in range(num_cols)] for _ in range(num_rows)]
    
    # Store cells with metadata
    cells_list = []
    cell_id = 0
    
    # Process each potential cell position
    for r in range(num_rows):
        for c in range(num_cols):
            # Skip if already processed
            if processed[r][c]:
                continue
            
            # Get starting corner
            top_left = grid[r][c]
            if top_left is None:
                continue
            
            # Detect cell span (how many columns and rows it occupies)
            col_span, row_span = detect_cell_span(gray, grid, r, c, num_rows, num_cols)
            
            # Get bottom-right corner based on span
            bottom_right = grid[r + row_span][c + col_span]
            
            if bottom_right is None:
                print(f"  Warning: Missing bottom-right corner for cell at r{r}_c{c}")
                processed[r][c] = True
                continue
            
            # Extract coordinates with padding
            x1 = int(top_left[0]) + padding
            y1 = int(top_left[1]) + padding
            x2 = int(bottom_right[0]) - padding
            y2 = int(bottom_right[1]) - padding
            
            # Validate bounds
            h, w = working_img.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                processed[r][c] = True
                continue
            
            # Extract cell image
            cell_img = working_img[y1:y2, x1:x2].copy()
            
            if cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                processed[r][c] = True
                continue
            
            # Create filename based on span
            if col_span > 1 or row_span > 1:
                cell_filename = f"cell_r{r}_c{c}_span{row_span}x{col_span}.png"
                span_info = f" (spans {row_span} rows × {col_span} cols)"
            else:
                cell_filename = f"cell_r{r}_c{c}.png"
                span_info = ""
            
            cell_path = os.path.join(output_dir, cell_filename)
            cv2.imwrite(cell_path, cell_img)
            
            # Store cell metadata
            cell_data = {
                "id": cell_id,
                "row": r,
                "col": c,
                "row_span": row_span,
                "col_span": col_span,
                "bbox": (x1, y1, x2, y2),
                "width": x2 - x1,
                "height": y2 - y1,
                "image": cell_img,
                "path": cell_path,
                "is_merged": col_span > 1 or row_span > 1
            }
            
            cells_list.append(cell_data)
            
            print(f"  Cell {cell_id}: r{r}_c{c}{span_info} → {cell_filename}")
            
            # Mark all grid positions occupied by this cell as processed
            for rr in range(r, r + row_span):
                for cc in range(c, c + col_span):
                    if rr < num_rows and cc < num_cols:
                        processed[rr][cc] = True
            
            cell_id += 1
    
    print(f"\n✓ Extracted {len(cells_list)} cells (including merged cells)")
    print(f"✓ Saved to: {output_dir}/")
    
    # Create grid map for visualization
    grid_map = create_grid_map(cells_list, num_rows, num_cols)
    
    return cells_list, grid_map


def detect_cell_span(gray_img, grid, start_row, start_col, num_rows, num_cols):
    """
    Detect if a cell spans multiple columns or rows by checking for missing internal lines.
    
    Returns:
    - (col_span, row_span): Number of columns and rows the cell spans
    """
    col_span = 1
    row_span = 1
    
    # === DETECT COLUMN SPAN (horizontal merge) ===
    # Check if there's a vertical line to the right
    for c in range(start_col + 1, num_cols + 1):
        top_left = grid[start_row][start_col]
        top_right = grid[start_row][c] if c < len(grid[start_row]) else None
        bottom_left = grid[start_row + 1][start_col] if start_row + 1 < len(grid) else None
        bottom_right = grid[start_row + 1][c] if (start_row + 1 < len(grid) and c < len(grid[start_row + 1])) else None
        
        if None in [top_left, top_right, bottom_left, bottom_right]:
            break
        
        # Check if vertical line exists between these columns
        x1 = int(top_right[0])
        y1 = int(top_right[1])
        y2 = int(bottom_right[1])
        
        if has_vertical_line_at_position(gray_img, x1, y1, y2):
            # Vertical line found, stop here
            break
        else:
            # No vertical line, cell continues
            col_span += 1
    
    # === DETECT ROW SPAN (vertical merge) ===
    # Check if there's a horizontal line below
    for r in range(start_row + 1, num_rows + 1):
        top_left = grid[start_row][start_col]
        bottom_left = grid[r][start_col] if r < len(grid) else None
        top_right = grid[start_row][start_col + 1] if start_col + 1 < len(grid[start_row]) else None
        bottom_right = grid[r][start_col + 1] if (r < len(grid) and start_col + 1 < len(grid[r])) else None
        
        if None in [top_left, bottom_left, top_right, bottom_right]:
            break
        
        # Check if horizontal line exists between these rows
        x1 = int(bottom_left[0])
        x2 = int(bottom_right[0])
        y1 = int(bottom_left[1])
        
        if has_horizontal_line_at_position(gray_img, x1, x2, y1):
            # Horizontal line found, stop here
            break
        else:
            # No horizontal line, cell continues
            row_span += 1
    
    return col_span, row_span


def has_vertical_line_at_position(gray_img, x, y1, y2, threshold=50, min_line_ratio=0.6):
    """
    Check if a vertical line exists at position x between y1 and y2.
    
    Returns True if at least min_line_ratio of pixels along the line are > threshold.
    """
    h, w = gray_img.shape
    
    if x < 0 or x >= w or y1 >= y2:
        return False
    
    y1 = max(0, int(y1))
    y2 = min(h, int(y2))
    
    # Check pixels along vertical line with some tolerance
    line_pixels = 0
    total_pixels = 0
    
    for y in range(y1, y2):
        for dx in range(-2, 3):  # Check ±2 pixels horizontally
            check_x = x + dx
            if 0 <= check_x < w:
                if gray_img[y, check_x] > threshold:
                    line_pixels += 1
                    break  # Found line pixel at this y position
        total_pixels += 1
    
    return (line_pixels / total_pixels) >= min_line_ratio if total_pixels > 0 else False


def has_horizontal_line_at_position(gray_img, x1, x2, y, threshold=50, min_line_ratio=0.6):
    """
    Check if a horizontal line exists at position y between x1 and x2.
    
    Returns True if at least min_line_ratio of pixels along the line are > threshold.
    """
    h, w = gray_img.shape
    
    if y < 0 or y >= h or x1 >= x2:
        return False
    
    x1 = max(0, int(x1))
    x2 = min(w, int(x2))
    
    # Check pixels along horizontal line with some tolerance
    line_pixels = 0
    total_pixels = 0
    
    for x in range(x1, x2):
        for dy in range(-2, 3):  # Check ±2 pixels vertically
            check_y = y + dy
            if 0 <= check_y < h:
                if gray_img[check_y, x] > threshold:
                    line_pixels += 1
                    break  # Found line pixel at this x position
        total_pixels += 1
    
    return (line_pixels / total_pixels) >= min_line_ratio if total_pixels > 0 else False


def create_grid_map(cells_list, num_rows, num_cols):
    """
    Create a 2D map showing which cell ID occupies each grid position.
    Useful for CSV reconstruction.
    
    Returns:
    - 2D list where grid_map[r][c] = cell_id or -1 if empty
    """
    grid_map = [[-1 for _ in range(num_cols)] for _ in range(num_rows)]
    
    for cell in cells_list:
        cell_id = cell["id"]
        r = cell["row"]
        c = cell["col"]
        row_span = cell["row_span"]
        col_span = cell["col_span"]
        
        # Fill all positions occupied by this cell
        for rr in range(r, r + row_span):
            for cc in range(c, c + col_span):
                if rr < num_rows and cc < num_cols:
                    grid_map[rr][cc] = cell_id
    
    return grid_map


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
    # cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "vert_mask.png"), vert_mask)
    
    
    # # Show results in windows
    # cv2.imshow("Original", img)
    # cv2.imshow("Horizontal Lines", horiz_mask)
    # cv2.imshow("Vertical Lines", vert_mask)
    

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    intersections = cv2.bitwise_and(horiz_mask, vert_mask)
    cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "intersections.png"), intersections)
    
   
    # найти центры connectedComponents
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(intersections, connectivity=8)
    # # centroids — список точек (x,y)
    # print("centroids:", centroids)
    # Make a copy so we don't modify original
    img_with_points = working_img.copy()

    # Draw each centroid as a small red circle
    for cx, cy in centroids:
        cv2.circle(img_with_points, (int(cx), int(cy)), radius=3, color=(0, 0, 255), thickness=-1)  # Red dot
        
    cv2.imwrite("steps_out/intersections_visualized.png", img_with_points)

        
    # rows_pts, viz_path, info = group_and_visualize_nodes(centroids, working_img, area_thresh=0, row_tol=None)
    rows_pts, viz_path, info = group_and_visualize_nodes(
        centroids=centroids,
        original_img=working_img,
        row_tol=8,
        trim_to_min_cols=True,
        out_path="steps_out/intersections_grouped.png"
    )
    print("Saved debug v gisualization to:", viz_path)
    print("Detected rows x cols:", len(rows_pts), len(rows_pts[0]) if rows_pts else 0)
    
    centroids_np = np.array(centroids)  # from your earlier step
    rows_of_points = build_rows_from_centroids(centroids_np, working_img.shape,
                                          area_thresh=0, row_tol=None,
                                          trim_to_min=True)

    if len(rows_of_points) < 2 or len(rows_of_points[0]) < 2:
        raise RuntimeError("Not enough grid nodes to extract cells. Try lowering area_thresh, increasing row_tol, or improving line masks.")

    # optional visualize nodes on the warped image:
    viz_path = os.path.join("steps_out", "grid_nodes_and_cells.png")
    os.makedirs("steps_out", exist_ok=True)
    visualize_grid_and_cells(rows_of_points, working_img, viz_path)
    print("Saved visualization:", viz_path)
    
    
    # === USAGE ===
    cells_list, grid_map = extract_table_cells_with_merge_detection(
        working_img, centroids_np, rows_pts
    )

    # Print summary
    print("\n=== Extraction Summary ===")
    for cell in cells_list:
        merge_info = ""
        if cell["is_merged"]:
            merge_info = f" [MERGED: {cell['row_span']}×{cell['col_span']}]"
        print(f"Cell {cell['id']}: Row {cell['row']}, Col {cell['col']}{merge_info}")
        print(f"  Size: {cell['width']}×{cell['height']}px")
        print(f"  Path: {cell['path']}")

    # Create CSV structure (for later OCR integration)
    csv_structure = cells_to_csv_structure(cells_list, grid_map)

    # Visualize grid map
    print("\n=== Grid Map (Cell IDs) ===")
    for row in grid_map:
        print("  " + " ".join(f"{cell_id:3d}" if cell_id >= 0 else "  -" for cell_id in row))
        


    # Show or save result
    # cv2.imshow("Intersection Points", img_with_points)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Optional: Save image

        # добавим непересекающиеся линии (сумма масок) для Hough
    # combined = cv2.bitwise_or(horiz_mask, vert_mask)

    # получить линии
    # horiz_lines = lines_from_mask(horiz_mask, min_line_length=50)
    # vert_lines = lines_from_mask(vert_mask, min_line_length=50)
    # cv2.imwrite(os.path.join(DEVELOPMENT_DIR, "lines_after.png"), cv2.bitwise_or(horiz_lines, vert_lines))

    # фильтруем почти горизонтальные/вертикальные по наклону
    # def is_horizontal(l, angle_thresh=10):
    #     x1,y1,x2,y2 = l
    #     dx = x2-x1
    #     dy = y2-y1
    #     if dx == 0:
    #         return False
    #     ang = abs(np.degrees(np.arctan2(dy, dx)))
    #     return ang < angle_thresh

    # def is_vertical(l, angle_thresh=10):
    #     x1,y1,x2,y2 = l
    #     dx = x2-x1
    #     dy = y2-y1
    #     if dx == 0:
    #         return True
    #     ang = abs(np.degrees(np.arctan2(dy, dx)))
    #     return abs(ang - 90) < angle_thresh

    # horiz_lines = [l for l in horiz_lines if is_horizontal(l)]
    # vert_lines = [l for l in vert_lines if is_vertical(l)]

    # # объединить близкие линии
    # horiz_merged = merge_similar_lines(horiz_lines, axis=1, thresh=LINE_MERGE_THRESH)
    # vert_merged = merge_similar_lines(vert_lines, axis=0, thresh=LINE_MERGE_THRESH)

    # # пересечения
    # points = []
    # for hl in horiz_merged:
    #     for vl in vert_merged:
    #         pt = line_intersection(hl, vl)
    #         if pt is not None:
    #             x,y = pt
    #             # проверить в пределах изображения
    #             H,W = combined.shape[:2]
    #             if 0 <= x < W and 0 <= y < H:
    #                 points.append((x,y))

    # if len(points) == 0:
    #     raise RuntimeError("Не удалось найти пересечений линий. Попробуйте изменить параметры предобработки/морфологии.")

    # # упорядочим точки в сетку
    # rows_of_points = sort_points_by_row_col(centroids, row_thresh=POINT_ROW_GROUP_THRESH)

    # # попытка привести все строки к одинаковой длине (у некоторых рядов может быть лишняя/меньше точек)
    # # берем минимальную длину
    # min_len = min(len(r) for r in rows_of_points)
    # rows_of_points = [r[:min_len] for r in rows_of_points]

    # # конвертируем к numpy arrays
    # rows_np = [np.array(r, dtype='float32') for r in rows_of_points]

    # # извлечь ячейки
    # cells = extract_cells_from_points(rows_np, working_img, out_dir=OUTPUT_DIR)

    # # подготовим координаты в формате [(row,col,[(x,y) tl,tr,br,bl]), ...] и вернём
    # coords = []
    # for c in cells:
    #     coords.append({'row': c['row'], 'col': c['col'], 'coords': c['coords'], 'path': c['path']})

    # # return {'cells': coords, 'rows': len(rows_np)-1, 'cols': min_len-1, 'warped': working_img, 'masks': (horiz_mask, vert_mask), 'merged_lines': (horiz_merged, vert_merged)}
    # return {'cells': coords, 'rows': len(rows_np)-1, 'cols': min_len-1, 'warped': working_img, 'masks': (horiz_mask, vert_mask)}
# ---------- Запуск ----------
if __name__ == "__main__":
    process_table_image(INPUT_PATH)
    # print(f"Найдено ячеек: {len(result['cells'])} ({result['rows']} rows x {result['cols']} cols)")
    # for c in result['cells'][:10]:
    #     print(f"r{c['row']} c{c['col']} -> {c['path']}")
    print("Готово. Ячейки сохранены в:", OUTPUT_DIR)


# ALL ACTUAL BEFORE GROUP_AND_VISUALIZE_NODES FUNCTION
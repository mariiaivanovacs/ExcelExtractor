import cv2
import numpy as np

def detect_and_draw_table_lines(working_img, centroids, rows_pts, 
                                threshold=50, h_check_len=30, v_check_len=6):
    """
    Detects and draws table lines between intersection points.
    
    Parameters:
    - working_img: Image to draw lines on
    - centroids: Array of all intersection points [(x, y), ...]
    - rows_pts: Grouped centroids organized by rows
    - threshold: Min pixel intensity to consider as line (default: 50)
    - h_check_len: Horizontal pixels to check for continuity (default: 6)
    - v_check_len: Vertical pixels to check for continuity (default: 6)
    
    Returns:
    - Image with detected lines drawn
    """
    result_img = working_img.copy()
    
    # Convert to grayscale if needed
    if len(working_img.shape) == 3:
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = working_img.copy()
    
    
    lines_drawn = 0
    
    # === STEP 1: HORIZONTAL LINES (Row by Row) ===
    print(f"\nProcessing horizontal lines across {len(rows_pts)} rows...")
    
    for row_idx, row in enumerate(rows_pts):
        if len(row) < 2:
            continue
        
        # Sort points in row by x-coordinate (left to right)
        sorted_row = sorted(row, key=lambda p: p[0])
        print(f"  Row {row_idx + 1} has {len(sorted_row)} points")
        
        for i in range(len(sorted_row) - 1):
            x1, y1 = int(sorted_row[i][0]), int(sorted_row[i][1])
            x2, y2 = int(sorted_row[i + 1][0]), int(sorted_row[i + 1][1])
            
            # Check if horizontal line exists between these intersections
            if check_horizontal_line(gray, x1, y1, x2, y2, threshold, h_check_len):
                cv2.line(result_img, (x1, y1), (x2, y2), (0,0,0), 1)
                lines_drawn += 1
    
    print(f"  Drew {lines_drawn} horizontal lines")
    
    # === STEP 2: VERTICAL LINES (Column by Column) ===
    print(f"\nProcessing vertical lines...")
    
    # Group centroids into columns
    cols_pts = group_centroids_by_columns(centroids, col_tol=4)
    print(f"  Found {len(cols_pts)} columns")
    
    v_lines_drawn = 0
    for col_idx, col in enumerate(cols_pts):
        if len(col) < 2:
            continue
        
        # Sort points in column by y-coordinate (top to bottom)
        sorted_col = sorted(col, key=lambda p: p[1])
        
        for i in range(len(sorted_col) - 1):
            x1, y1 = int(sorted_col[i][0]), int(sorted_col[i][1])
            x2, y2 = int(sorted_col[i + 1][0]), int(sorted_col[i + 1][1])
            
            # Check if vertical line exists between these intersections
            if check_vertical_line(gray, x1, y1, x2, y2, threshold, v_check_len):
                cv2.line(result_img, (x1, y1), (x2, y2), (0,0,0), 1)
                v_lines_drawn += 1
    
    print(f"  Drew {v_lines_drawn} vertical lines")
    print(f"\nTotal lines drawn: {lines_drawn + v_lines_drawn}")
    
    return result_img


def check_horizontal_line(gray_img, x1, y1, x2, y2, threshold=50, check_len=6):
    """
    Check if horizontal line exists between two points.
    Looks ±3 pixels vertically for 5-6 consecutive pixels > threshold.
    """
    h, w = gray_img.shape
    
    # Need at least some distance between points
    if x2 - x1 < 5:
        return False
    
    # Sample multiple positions along the expected line
    num_checks = min(3, (x2 - x1) // 10 + 1)
    checks_passed = 0
    
    for sample_idx in range(num_checks):
        # Sample x position between the two points
        if num_checks == 1:
            x_check = (x1 + x2) // 2
        else:
            x_check = x1 + int((x2 - x1) * (sample_idx + 1) / (num_checks + 1))
        
        if x_check < 0 or x_check >= w - check_len:
            continue
        
        # Check ±3 pixels vertically around expected y position
        line_found = False
        for dy in range(-3, 4):
            y_check = y1 + dy
            if y_check < 0 or y_check >= h:
                continue
            
            # Count consecutive pixels horizontally
            consecutive = 0
            for dx in range(check_len):
                if x_check + dx >= w:
                    break
                if gray_img[y_check, x_check + dx] > threshold:
                    consecutive += 1
                    if consecutive >= 5:  # Found 5+ consecutive pixels
                        line_found = True
                        break
                else:
                    consecutive = 0
            
            if line_found:
                break
        
        if line_found:
            checks_passed += 1
    
    # Return True if majority of sample points show a line
    return checks_passed >= (num_checks + 1) // 2


def check_vertical_line(gray_img, x1, y1, x2, y2, threshold=50, check_len=6):
    """
    Check if vertical line exists between two points.
    Looks ±3 pixels horizontally for 5-6 consecutive pixels > threshold.
    """
    h, w = gray_img.shape
    
    # Need at least some distance between points
    if y2 - y1 < 5:
        return False
    
    # Sample multiple positions along the expected line
    num_checks = min(3, (y2 - y1) // 10 + 1)
    checks_passed = 0
    
    for sample_idx in range(num_checks):
        # Sample y position between the two points
        if num_checks == 1:
            y_check = (y1 + y2) // 2
        else:
            y_check = y1 + int((y2 - y1) * (sample_idx + 1) / (num_checks + 1))
        
        if y_check < 0 or y_check >= h - check_len:
            continue
        
        # Check ±3 pixels horizontally around expected x position
        line_found = False
        for dx in range(-3, 4):
            x_check = x1 + dx
            if x_check < 0 or x_check >= w:
                continue
            
            # Count consecutive pixels vertically
            consecutive = 0
            for dy in range(check_len):
                if y_check + dy >= h:
                    break
                if gray_img[y_check + dy, x_check] > threshold:
                    consecutive += 1
                    if consecutive >= 5:  # Found 5+ consecutive pixels
                        line_found = True
                        break
                else:
                    consecutive = 0
            
            if line_found:
                break
        
        if line_found:
            checks_passed += 1
    
    # Return True if majority of sample points show a line
    return checks_passed >= (num_checks + 1) // 2


def group_centroids_by_columns(centroids, col_tol=2):
    """
    Group centroids into columns based on x-coordinate proximity.
    """
    if centroids is None or len(centroids) == 0:
        return []
    
    # Sort by x-coordinate
    sorted_centroids = sorted(centroids, key=lambda p: p[0])
    
    columns = []
    current_col = [sorted_centroids[0]]
    
    for i in range(1, len(sorted_centroids)):
        # If x-coordinate is close to current column, add to it
        if abs(sorted_centroids[i][0] - current_col[0][0]) <= col_tol:
            current_col.append(sorted_centroids[i])
        else:
            # Start new column
            columns.append(current_col)
            current_col = [sorted_centroids[i]]
    
    # Add last column
    if current_col:
        columns.append(current_col)
    
    # for i in range(len(columns)):
    #     print(f"THE column {i} is : {columns[i]} and its length: {len(columns[i])}")
        
    return columns


# === USAGE ===

# NOT IN FINAL PIPELINE


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



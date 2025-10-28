import numpy as np
import matplotlib as plt 
import matplotlib.pyplot as plt

# --- Step 1: Create uneven grid data ---

# # Y coordinates for 10 rows (irregular spacing)
# y_positions = [0, 30, 70, 120, 180, 250, 330, 420, 520, 630]

# rows_sorted = []
# for y in y_positions:
#     # X coordinates along each row (irregular spacing)
#     x_positions = [0, 40, 100, 180, 280, 400, 550, 700, 900, 1100]
#     row_points = [(x, y) for x in x_positions]
#     rows_sorted.append(row_points)

# # X coordinates for 10 columns (irregular spacing)
# x_positions = [0, 60, 150, 260, 400, 550, 720, 900, 1100, 1300]

# cols_sorted = []
# for x in x_positions:
#     # Y coordinates along each column (same as row Y positions)
#     col_points = [(x, y) for y in y_positions]
#     cols_sorted.append(col_points)

# --- Step 2: Visualize with matplotlib ---

def visualize_code(rows_sorted, cols_sorted):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    # Draw rows (blue)
    for i, row in enumerate(rows_sorted):
        xs, ys = zip(*row)
        label = "Row" if i == 0 else ""
        plt.plot(xs, ys, 'b-o', alpha=0.6, label=label)

    # Draw columns (green)
    for j, col in enumerate(cols_sorted):
        xs, ys = zip(*col)
        label = "Column" if j == 0 else ""
        plt.plot(xs, ys, 'g-o', alpha=0.6, label=label)

    # plt.title("Visualization of Rows and Columns")
    # plt.xlabel("X coordinate")
    # plt.ylabel("Y coordinate")
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    # plt.gca().invert_yaxis()
    # plt.show()

    
    
# print(len(rows_sorted))
# print("-------------\n\n\n\n\n")
# print(len(cols_sorted))


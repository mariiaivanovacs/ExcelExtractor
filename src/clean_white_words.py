# i need to find the image in this folder the max width and max height of images 
import os
import cv2
import numpy as np

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
    
    h, w = img_float.shape
    top_30 = img_float[:int(h*0.3), :]
    bottom_30=img_float[-int(h*0.3):, :]
    middle_40 = img_float[int(h*0.3):int(h*0.7), :]

    
    
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
    
    return column_intensities
    

def clean_cells(path):
    # path = "blobs"
    # ensure cells_cleaned exists 
    os.makedirs(path, exist_ok=True)
    
    max_width = 0
    max_height = 0
    max_width_file = ""
    max_height_file = ""
    removed_names = []
    for filename in os.listdir(path):
        if filename.endswith('.png'):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # calculate column intensities 
            column_intensities = calculate_column_intensity(img, 1,)
            arr = np.asarray(column_intensities).ravel().astype(float)
            n = arr.size

            # Normalize to [0,1] if needed (e.g., values in 0..255)
            if arr.max() < 1.0:
                arr = arr / float(arr.max())

            threshold =0.8
            states = arr < 0.5
            pattern_line = 0
            start_of_pattern = 0
            max_line = 0
                
            for i, v in enumerate(states):

                if v:
                    # print("FOund black")
                    if start_of_pattern == 0:
                        start_of_pattern = 1
                    else:
                        start_of_pattern += 1
                else:
                    if start_of_pattern > 0:
                        # print("add pixel")
                        pattern_line += start_of_pattern
                        if start_of_pattern > max_line:
                            max_line = start_of_pattern
                        start_of_pattern = 0
                    else:
                        pattern_line -= 1
            if start_of_pattern > 0:
                pattern_line += start_of_pattern
                if start_of_pattern > max_line:
                    max_line = start_of_pattern
                    
            if pattern_line < len(arr)*0.3 and max_line <= 4:
                print("This is white cell")
                # print(f'States: {states}')
                # print(f"Arr: {arr}")
                # print(f"Pattern line: {pattern_line}, max_line: {max_line}")
                removed_names.append(filename)
                os.remove(img_path)
            elif max_line <= 2:
                print("This is white cell")
                # print(f"Pattern line: {pattern_line}, max_line: {max_line}")
                # print(f"Arr: {arr}")

                removed_names.append(filename)
                os.remove(img_path)
            else:
                print("This is not white cell")
                # save img
                # cv2.imwrite(f"cells_cleaned/{filename}", img)

    print(f"Removed {len(removed_names)} files")
    print(f"Files removed: {removed_names}")
    print(f"Amount of files left: {len(os.listdir(path))}")

            
                


    print(f"Max width: {max_width}")
    print(f"Max height: {max_height}")
    print(f"Max width file: {max_width_file}")
    print(f"Max height file: {max_height_file}")


    
def remove_black_borders(img: np.ndarray, side: int = 0, max_explored: int=0) -> np.ndarray:
    # number = 3
    stopped_number = -1
    totally_explored = 0
    limit = min(3, max_explored)
    have_white = 0
    # print("img shape: ", img.shape)
    # print(f"Side is: {side}")

    i = 0
    while i < limit:
        if side == 0:
            sequence = img[i,:]
        elif side == 1:
            sequence = img[:,img.shape[1]-i-1]
        elif side == 2:
            sequence = img[img.shape[0]-i-1,:]
        elif side == 3:
            sequence = img[:,i]
        average = np.sum(sequence) / len(sequence)

        
        range_to_check = len(sequence)
        pattern_line = 0
        start_of_pattern = 0
        max_line = 0
            
        for j in range(range_to_check):
            if sequence[j] >=230:
                have_white += 1
            if sequence[j] < 30:
                # print("FOund black")
                if start_of_pattern == 0:
                    start_of_pattern = 1
                else:
                    start_of_pattern += 1
            else:
                if start_of_pattern > 0:
                    # print("add pixel")
                    pattern_line += start_of_pattern
                    if start_of_pattern > max_line:
                        max_line = start_of_pattern
                    start_of_pattern = 0
                else:
                    pattern_line -= 1
        if start_of_pattern > 0:
            pattern_line += start_of_pattern
            if start_of_pattern > max_line:
                max_line = start_of_pattern
        
        
        white_pixels = have_white / len(sequence) * 100
        max_line = max_line / len(sequence) * 100
       
        if white_pixels < 10:
            # print("Found white border, add 2 more for")
            stopped_number = totally_explored
            limit = min(limit + 3, max_explored)
            # print(f"Current lim: {limit}")
            # for cells - 0.4 for blob - 0.75 (aspect ratio)
        if average < 125 or pattern_line > len(sequence)*0.75 or max_line >=25 :
            stopped_number = totally_explored
            limit = min(limit + 3, max_explored)
        elif average < 210 and pattern_line > len(sequence)*0.4:
            # print("Found black border, add 2 more for")
            stopped_number = totally_explored
            limit = min(limit + 3, max_explored)

        i += 1
        totally_explored += 1 
    # print(f"Stopped number: {stopped_number}")
    # print(f"Totally explored: {totally_explored}")
    
    return stopped_number
    
    
def remove_all_black_borders():
    os.makedirs("cells_cleaned", exist_ok=True)

    
    working_directory = "cells_production"
    
    # ensure cells_cleaned exists 

    for filename in os.listdir(working_directory):
        if filename.endswith('.png'):
            img = cv2.imread(f"{working_directory}/{filename}")
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            borders = [0,0,0,0]
            w,h = img.shape
            # print(f"Filename is: {filename}")
            # print(f"Width: {w}, Height: {h}")
            for i in range(4):
                # 0-top, 1-right, 2-bottom, 3-left
                borders[i] = remove_black_borders(img, i, min(w,h)-1)
                if borders[i] != -1:
                    # print(f"Cropping image, {i}")
                    if i==0:
                        img = img[borders[i]+1:,:]
                        w,h = img.shape

                    elif i==1:
                        # print("Cutting right")
                        # print(f"New width: {h-borders[i]-1}")
                        img = img[:,:h-borders[i]-1]
                        w,h = img.shape

                    elif i==2:
                        img = img[:w - borders[i]-1,:]
                        w,h = img.shape
            
                    elif i==3:
                        img = img[:,borders[i]+1:]
                        w,h = img.shape

                img = img

            cv2.imwrite(f"cells_cleaned/{filename}", img)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image processing pipeline step.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["blobs", "words", "remove_border", "clean_cells"],
        help="Choose which processing step to run."
    )
    parser.add_argument(
        "--folder",
        required=False,
        default="cells_production",
        help="Choose which folder to run on."
    )
    args = parser.parse_args()

    mode = args.mode
    folder = args.folder
    if mode=="remove_border":
        remove_all_black_borders()
    elif mode == "clean_cells":
        clean_cells(folder)

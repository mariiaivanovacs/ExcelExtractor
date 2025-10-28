path = "words_production"

# i need to find the image in this folder the max width and max height of images 
import os
import cv2
# save the name of the file the max width and max height are achieved
max_width = 0
max_height = 0
max_width_file = ""
max_height_file = ""
heights = {}
widths = []
for filename in os.listdir(path):
    if filename.endswith('.png'):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            if filename.startswith("debug_"):
                continue
            else:
                height, width, _ = img.shape
                if height not in heights:
                    heights[height] = 1
                else:
                    heights[height] += 1
                widths.append(width)
                
                if width > max_width:
                    max_width = width
                    max_width_file = filename
                if height > max_height:
                    max_height = height
                    max_height_file = filename

print(f"Max width: {max_width}")
print(f"Max height: {max_height}")
print(f"Max width file: {max_width_file}")
print(f"Max height file: {max_height_file}")

print(f"Height distribution: {heights}")
print(f"Average width: {sum(widths)/len(widths)}")

print(f"Amount of files in folder: {len(os.listdir(path))}")
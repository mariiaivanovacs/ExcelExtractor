import cv2
import os
import numpy as np

# Assume your preprocess and find_table_contour functions are defined above

# Load an image (replace with your own test image path)
img_path = "data/input/original.jpeg"  # e.g., photo of a receipt, table, document
img = cv2.imread(img_path)
DEVELOPMENT_DIR = "steps_out"


# ---------- Основные шаги ----------
# ---------- Основные шаги ----------

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
    bright_thresh = 0.86
    color_thresh = 0.1

    # Mask for light gray pixels
    mask = (brightness > bright_thresh) & (color_diff < color_thresh)

    # Replace with white
    img[mask] = [1.0, 1.0, 1.0]

    # Convert back to uint8
    result = (img * 255).astype(np.uint8)
    cv2.imwrite(f"steps_out/{output_path}", result)
    return result

def preprocess(img):
    filename_1 = "grayscale_image.png"
    filename_2 = "thresholded_image.png"
    filename_3 = "denoised_image.png"
    path_1 = os.path.join(DEVELOPMENT_DIR, filename_1)
    path_2 = os.path.join(DEVELOPMENT_DIR, filename_2)
    path_3 = os.path.join(DEVELOPMENT_DIR, filename_3)
    
    
    gray = mask_pixels(img, "mask_1.png")
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
    gray = clahe.apply(gray)
    gray_2 = mask_pixels(img, "mask_2.png")
    den = cv2.bilateralFilter(gray_2, d=9, sigmaColor=30, sigmaSpace=30)
    
    
    # Ensure single-channel uint8 before thresholding
    if len(den.shape) == 3:
        den = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)

    if den.dtype != np.uint8:
        den = (den * 255).astype(np.uint8)
    # # adaptive threshold - хорош для разного освещения
    th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
   
    cv2.imwrite(path_1, gray)
    cv2.imwrite(path_2, th)
    cv2.imwrite(path_3, den)
    return gray, th


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
            return approx.reshape(4,2)
    return None

if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

# Preprocess to get thresholded image
gray, thresh = preprocess(img)  # your existing function

# Find table contour
table_contour = find_table_contour(thresh)

# Draw result
output = img.copy()

if table_contour is not None:
    print("✅ Table contour found!")
    print("Corners:", table_contour)
    
    # Draw the contour (in green)
    cv2.drawContours(output, [table_contour], -1, (0, 255, 0), 3)
    
    # Optionally: label each corner
    for i, (x, y) in enumerate(table_contour):
        cv2.circle(output, (x, y), 8, (0, 0, 255), -1)  # red dot
        cv2.putText(output, f"P{i}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
else:
    print("❌ No table contour found.")
OUTPUT_DIR = "steps_out"
path_1 = os.path.join(OUTPUT_DIR, "output_with_contour.jpg")
cv2.imwrite(path_1, output)
path_2 = os.path.join(OUTPUT_DIR, "threshold_used.jpg")
cv2.imwrite(path_2, thresh)
# Save or show results

# Optional: display (if running locally)
# cv2.imshow("Original", img)
# cv2.imshow("Threshold", thresh)
# cv2.imshow("Detected Contour", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
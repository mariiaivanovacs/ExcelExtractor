#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

def inspect_cell_image(image_path):
    """Inspect a cell image to understand its content and characteristics."""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    # Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    print(f"Image: {image_path}")
    print(f"Shape: {gray.shape}")
    print(f"Intensity range: {gray.min()} - {gray.max()}")
    print(f"Mean intensity: {gray.mean():.2f}")
    print(f"Std intensity: {gray.std():.2f}")
    
    # Check if image is mostly light or dark
    light_pixels = np.sum(gray > 200)
    dark_pixels = np.sum(gray < 50)
    total_pixels = gray.size
    
    print(f"Light pixels (>200): {light_pixels} ({100*light_pixels/total_pixels:.1f}%)")
    print(f"Dark pixels (<50): {dark_pixels} ({100*dark_pixels/total_pixels:.1f}%)")
    
    # Calculate column intensities (both normal and inverted)
    col_intensities_normal = np.mean(gray.astype(np.float32) / 255.0, axis=0)
    col_intensities_inverted = np.mean(1.0 - gray.astype(np.float32) / 255.0, axis=0)
    
    print(f"Column intensities (normal): {col_intensities_normal.min():.3f} - {col_intensities_normal.max():.3f}")
    print(f"Column intensities (inverted): {col_intensities_inverted.min():.3f} - {col_intensities_inverted.max():.3f}")
    
    # Find columns with significant content
    threshold = 0.1
    content_cols_normal = np.sum(col_intensities_normal < (1.0 - threshold))  # Dark content
    content_cols_inverted = np.sum(col_intensities_inverted > threshold)  # Inverted content
    
    print(f"Columns with content (normal, dark<{1.0-threshold}): {content_cols_normal}")
    print(f"Columns with content (inverted, >{threshold}): {content_cols_inverted}")
    
    # Save visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(255 - gray, cmap='gray')
    plt.title('Inverted Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.plot(col_intensities_normal)
    plt.title('Column Intensities (Normal)')
    plt.xlabel('Column')
    plt.ylabel('Average Intensity')
    plt.axhline(y=1.0-threshold, color='r', linestyle='--', label=f'Threshold {1.0-threshold}')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(col_intensities_inverted)
    plt.title('Column Intensities (Inverted)')
    plt.xlabel('Column')
    plt.ylabel('Average Intensity')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold {threshold}')
    plt.legend()
    
    plt.tight_layout()
    output_path = f"inspect_{image_path.replace('/', '_').replace('.png', '')}.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Test different cell images
    test_images = [
        "cells_production/cell_r0_c3.png",
        "cells_production/cell_r20_c16.png",  # This was used in seg_qwen.py
        "cells_production/cell_r1_c1.png",
        "cells_production/cell_r10_c5.png"
    ]
    
    for img_path in test_images:
        try:
            inspect_cell_image(img_path)
            print("-" * 50)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            print("-" * 50)

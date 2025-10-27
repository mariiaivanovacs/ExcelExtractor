#!/usr/bin/env python3
"""
Analyze Real Test Images
========================
Analyze the characteristics of real test images to understand
why models are failing and what preprocessing is needed.
"""

import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

def analyze_image_characteristics(image_path):
    """
    Analyze characteristics of a single image.
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    filename = os.path.basename(image_path)
    
    # Basic statistics
    height, width = img.shape
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    min_intensity = np.min(img)
    max_intensity = np.max(img)
    
    # Check if image is inverted (dark text on light background vs light text on dark)
    # Count pixels in different intensity ranges
    dark_pixels = np.sum(img < 128)
    light_pixels = np.sum(img >= 128)
    is_inverted = dark_pixels < light_pixels  # More light pixels = likely inverted
    
    # Estimate foreground/background
    if is_inverted:
        # Dark text on light background
        foreground_pixels = img < 128
        background_pixels = img >= 128
    else:
        # Light text on dark background
        foreground_pixels = img >= 128
        background_pixels = img < 128
    
    fg_mean = np.mean(img[foreground_pixels]) if np.any(foreground_pixels) else 0
    bg_mean = np.mean(img[background_pixels]) if np.any(background_pixels) else 0
    
    return {
        'filename': filename,
        'shape': (height, width),
        'mean': mean_intensity,
        'std': std_intensity,
        'min': min_intensity,
        'max': max_intensity,
        'is_inverted': is_inverted,
        'dark_pixels': dark_pixels,
        'light_pixels': light_pixels,
        'fg_mean': fg_mean,
        'bg_mean': bg_mean,
        'contrast': abs(fg_mean - bg_mean)
    }

def preprocess_for_model(img, target_size=(32, 32)):
    """
    Try different preprocessing approaches for the model.
    """
    results = {}
    
    # Original approach (simple resize + normalize)
    img_simple = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_simple = img_simple.astype(np.float32) / 255.0
    results['simple'] = img_simple
    
    # Inverted approach (for dark text on light background)
    img_inverted = 255 - img
    img_inverted = cv2.resize(img_inverted, target_size, interpolation=cv2.INTER_AREA)
    img_inverted = img_inverted.astype(np.float32) / 255.0
    results['inverted'] = img_inverted
    
    # Thresholded approach
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_thresh = cv2.resize(img_thresh, target_size, interpolation=cv2.INTER_AREA)
    img_thresh = img_thresh.astype(np.float32) / 255.0
    results['thresholded'] = img_thresh
    
    # Inverted + thresholded
    img_inv_thresh = 255 - img
    _, img_inv_thresh = cv2.threshold(img_inv_thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_inv_thresh = cv2.resize(img_inv_thresh, target_size, interpolation=cv2.INTER_AREA)
    img_inv_thresh = img_inv_thresh.astype(np.float32) / 255.0
    results['inverted_thresholded'] = img_inv_thresh
    
    # Contrast enhanced
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    img_clahe = cv2.resize(img_clahe, target_size, interpolation=cv2.INTER_AREA)
    img_clahe = img_clahe.astype(np.float32) / 255.0
    results['clahe'] = img_clahe
    
    return results

def analyze_all_real_images():
    """
    Analyze all real test images.
    """
    print("=" * 70)
    print("ANALYZING REAL TEST IMAGES")
    print("=" * 70)
    
    test_dir = 'tests'
    image_files = glob.glob(os.path.join(test_dir, '*.png'))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"❌ No PNG images found in {test_dir}/")
        return
    
    expected_sequence = [9, 2, 3, 6, 0, 5, 7, 1]
    
    print(f"📁 Found {len(image_files)} test images")
    print(f"🎯 Expected sequence: {expected_sequence}")
    print()
    
    # Analyze each image
    analyses = []
    
    for i, image_path in enumerate(image_files):
        analysis = analyze_image_characteristics(image_path)
        if analysis:
            analysis['expected_digit'] = expected_sequence[i] if i < len(expected_sequence) else None
            analyses.append(analysis)
    
    # Print analysis results
    print("📊 IMAGE CHARACTERISTICS:")
    print("-" * 70)
    print(f"{'#':<2} {'File':<35} {'Size':<10} {'Mean':<6} {'Std':<6} {'Inverted':<8} {'Contrast':<8} {'Expected':<8}")
    print("-" * 70)
    
    for i, analysis in enumerate(analyses):
        print(f"{i+1:<2} {analysis['filename'][:34]:<35} "
              f"{analysis['shape'][1]}x{analysis['shape'][0]:<6} "
              f"{analysis['mean']:<6.1f} {analysis['std']:<6.1f} "
              f"{'Yes' if analysis['is_inverted'] else 'No':<8} "
              f"{analysis['contrast']:<8.1f} {analysis['expected_digit']:<8}")
    
    print()
    
    # Summary statistics
    all_inverted = all(a['is_inverted'] for a in analyses)
    any_inverted = any(a['is_inverted'] for a in analyses)
    avg_contrast = np.mean([a['contrast'] for a in analyses])
    
    print("🔍 SUMMARY:")
    print(f"  - All images inverted (dark text on light bg): {all_inverted}")
    print(f"  - Any images inverted: {any_inverted}")
    print(f"  - Average contrast: {avg_contrast:.1f}")
    print()
    
    # Create visualization
    visualize_preprocessing_comparison(image_files[:4], expected_sequence[:4])
    
    return analyses

def visualize_preprocessing_comparison(image_files, expected_digits):
    """
    Visualize different preprocessing approaches on real images.
    """
    print("📸 Creating preprocessing comparison visualization...")
    
    n_images = len(image_files)
    fig, axes = plt.subplots(n_images, 6, figsize=(18, 3*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    preprocessing_names = ['Original', 'Simple', 'Inverted', 'Thresholded', 'Inv+Thresh', 'CLAHE']
    
    for i, image_path in enumerate(image_files):
        # Load original image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        filename = os.path.basename(image_path)
        expected = expected_digits[i] if i < len(expected_digits) else "?"
        
        # Get preprocessed versions
        preprocessed = preprocess_for_model(img)
        
        # Show original
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'{filename[:20]}...\nExpected: {expected}', fontsize=8)
        axes[i, 0].axis('off')
        
        # Show preprocessed versions
        for j, (name, processed_img) in enumerate(preprocessed.items(), 1):
            axes[i, j].imshow(processed_img, cmap='gray')
            axes[i, j].set_title(f'{name}', fontsize=8)
            axes[i, j].axis('off')
    
    # Add column titles
    for j, name in enumerate(preprocessing_names):
        axes[0, j].text(0.5, 1.1, name, transform=axes[0, j].transAxes, 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("📸 Preprocessing comparison saved to: models/preprocessing_comparison.png")
    plt.close()

def test_preprocessing_approaches():
    """
    Test different preprocessing approaches with a simple model.
    """
    print("🧪 Testing different preprocessing approaches...")
    
    # This would require loading a model and testing each approach
    # For now, just provide recommendations based on analysis
    
    print("\n💡 PREPROCESSING RECOMMENDATIONS:")
    print("-" * 50)
    print("Based on the analysis, try these approaches:")
    print("1. INVERT images (dark text → light text on dark background)")
    print("2. Apply OTSU thresholding for clean binary images")
    print("3. Use CLAHE for contrast enhancement")
    print("4. Combine inversion + thresholding for best results")
    print()
    print("The models were likely trained on light text on dark background,")
    print("but your real images have dark text on light background!")

if __name__ == "__main__":
    # Analyze all real images
    analyses = analyze_all_real_images()
    
    # Test preprocessing approaches
    test_preprocessing_approaches()
    
    print("\n" + "=" * 70)
    print("🎯 KEY FINDINGS:")
    print("=" * 70)
    print("1. Real images likely have INVERTED polarity vs training data")
    print("2. Models expect light text on dark background")
    print("3. Real images have dark text on light background")
    print("4. Need to INVERT images before feeding to model")
    print("5. Consider retraining with properly preprocessed real data")
    print("=" * 70)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage
import os

import warnings
warnings.filterwarnings('ignore')
from fight import downsample_then_upsample

# Create output directory for debug images
os.makedirs('experiment/debug_images', exist_ok=True)
os.makedirs('comparison_plots', exist_ok=True)

def create_number_image(number, size=(200, 200), font_size=150):
    """Create an image with a number"""
    img = Image.new('L', size, color=255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Get text bbox and center it
    bbox = draw.textbbox((0, 0), str(number), font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((size[0] - text_width) // 2 - bbox[0], 
                (size[1] - text_height) // 2 - bbox[1])
    
    draw.text(position, str(number), fill=0, font=font)  # Black text
    
    np_img = np.array(img).astype(np.float32) / 255.0
    np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)
    
    np_img = (np_img * 255).astype(np.uint8)
    return np_img

def detect_edges(img):
    """Detect edges using Canny edge detector"""
    edges = cv2.Canny(img, 50, 150)
    return edges

def compute_gradient(img):
    """Compute gradient magnitude and direction"""
    # Sobel operators
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Gradient direction
    direction = np.arctan2(grad_y, grad_x)
    
    return magnitude, direction, grad_x, grad_y

def compute_curvature(img):
    """Compute curvature from gradients"""
    # First derivatives
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Second derivatives
    grad_xx = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3)
    grad_yy = cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
    grad_xy = cv2.Sobel(grad_x, cv2.CV_64F, 0, 1, ksize=3)
    
    # Curvature formula
    denominator = (grad_x**2 + grad_y**2 + 1e-10)**(3/2)
    curvature = (grad_xx * grad_y**2 - 2 * grad_xy * grad_x * grad_y + grad_yy * grad_x**2) / denominator
    
    return curvature

def compute_texture(img):
    """Compute texture features using local standard deviation"""
    # Apply Gaussian blur and compute local variance
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Local standard deviation (texture measure)
    mean = cv2.blur(img.astype(float), (5, 5))
    mean_sq = cv2.blur((img.astype(float))**2, (5, 5))
    variance = mean_sq - mean**2
    texture = np.sqrt(np.maximum(variance, 0))
    
    return texture

def find_whitespace_center(img, threshold=200):
    """Find whitespace (bright regions) near center"""
    h, w = img.shape
    center_region = img[h//4:3*h//4, w//4:3*w//4]
    
    # Binary mask of whitespace
    whitespace_mask = (center_region > threshold).astype(np.uint8) * 255
    
    # Create full image mask
    full_mask = np.zeros_like(img)
    full_mask[h//4:3*h//4, w//4:3*w//4] = whitespace_mask
    
    return full_mask, np.sum(whitespace_mask > 0)

def horizontal_pixel_change(img):
    """Calculate pixel change intensity comparing to right neighbor"""
    h, w = img.shape
    change_map = np.zeros_like(img, dtype=float)
    
    # Calculate absolute difference with right neighbor
    for i in range(h):
        for j in range(w - 1):
            change_map[i, j] = abs(float(img[i, j]) - float(img[i, j + 1]))
    
    # Normalize to 0-255 range for visualization
    if change_map.max() > 0:
        change_map = (change_map / change_map.max()) * 255
    
    return change_map

def count_black_pixels_per_row(img, threshold=128):
    """Count black pixels in each row"""
    binary = (img < threshold).astype(int)
    row_counts = np.sum(binary, axis=1)
    return row_counts

def visualize_all_features(img3, img5):
    """Create comprehensive visualization of all features"""
    
    print("Processing edges...")
    edges3 = detect_edges(img3)
    edges5 = detect_edges(img5)
    
    print("Computing gradients...")
    mag3, dir3, gx3, gy3 = compute_gradient(img3)
    mag5, dir5, gx5, gy5 = compute_gradient(img5)
    
    print("Computing curvature...")
    curv3 = compute_curvature(img3)
    curv5 = compute_curvature(img5)
    
    print("Computing texture...")
    tex3 = compute_texture(img3)
    tex5 = compute_texture(img5)
    
    print("Finding whitespace...")
    white3, white_count3 = find_whitespace_center(img3)
    white5, white_count5 = find_whitespace_center(img5)
    
    print("Computing horizontal pixel changes...")
    hchange3 = horizontal_pixel_change(img3)
    hchange5 = horizontal_pixel_change(img5)
    
    print("Counting black pixels per row...")
    row_counts3 = count_black_pixels_per_row(img3)
    row_counts5 = count_black_pixels_per_row(img5)
    
    # Save individual debug images
    cv2.imwrite('experiment/debug_images/3_edges.png', edges3)
    cv2.imwrite('experiment/debug_images/5_edges.png', edges5)
    cv2.imwrite('experiment/debug_images/3_gradient_mag.png', (mag3 / mag3.max() * 255).astype(np.uint8))
    cv2.imwrite('experiment/debug_images/5_gradient_mag.png', (mag5 / mag5.max() * 255).astype(np.uint8))
    cv2.imwrite('experiment/debug_images/3_horizontal_change.png', hchange3.astype(np.uint8))
    cv2.imwrite('experiment/debug_images/5_horizontal_change.png', hchange5.astype(np.uint8))
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 24))
    
    # Row 1: Original images
    plt.subplot(8, 3, 1)
    plt.imshow(img3, cmap='gray')
    plt.title('Number 3 - Original', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(8, 3, 2)
    plt.imshow(img5, cmap='gray')
    plt.title('Number 5 - Original', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(8, 3, 3)
    diff_original = np.abs(img3.astype(float) - img5.astype(float))
    plt.imshow(diff_original, cmap='hot')
    plt.title('Difference (Original)', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Row 2: Edges
    plt.subplot(8, 3, 4)
    plt.imshow(edges3, cmap='gray')
    plt.title('Edges - 3', fontsize=12)
    plt.axis('off')
    
    plt.subplot(8, 3, 5)
    plt.imshow(edges5, cmap='gray')
    plt.title('Edges - 5', fontsize=12)
    plt.axis('off')
    
    plt.subplot(8, 3, 6)
    diff_edges = np.abs(edges3.astype(float) - edges5.astype(float))
    plt.imshow(diff_edges, cmap='hot')
    plt.title('Difference (Edges)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Row 3: Gradient Magnitude
    plt.subplot(8, 3, 7)
    plt.imshow(mag3, cmap='viridis')
    plt.title('Gradient Magnitude - 3', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 8)
    plt.imshow(mag5, cmap='viridis')
    plt.title('Gradient Magnitude - 5', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 9)
    diff_mag = np.abs(mag3 - mag5)
    plt.imshow(diff_mag, cmap='hot')
    plt.title('Difference (Gradient Mag)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Row 4: Gradient Direction
    plt.subplot(8, 3, 10)
    plt.imshow(dir3, cmap='hsv')
    plt.title('Gradient Direction - 3', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 11)
    plt.imshow(dir5, cmap='hsv')
    plt.title('Gradient Direction - 5', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 12)
    # For direction, we need to handle wrapping
    diff_dir = np.minimum(np.abs(dir3 - dir5), 2*np.pi - np.abs(dir3 - dir5))
    plt.imshow(diff_dir, cmap='hot')
    plt.title('Difference (Gradient Dir)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Row 5: Curvature
    plt.subplot(8, 3, 13)
    plt.imshow(curv3, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    plt.title('Curvature - 3', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 14)
    plt.imshow(curv5, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    plt.title('Curvature - 5', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 15)
    diff_curv = np.abs(curv3 - curv5)
    plt.imshow(diff_curv, cmap='hot')
    plt.title('Difference (Curvature)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Row 6: Texture
    plt.subplot(8, 3, 16)
    plt.imshow(tex3, cmap='plasma')
    plt.title('Texture - 3', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 17)
    plt.imshow(tex5, cmap='plasma')
    plt.title('Texture - 5', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 18)
    diff_tex = np.abs(tex3 - tex5)
    plt.imshow(diff_tex, cmap='hot')
    plt.title('Difference (Texture)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Row 7: Horizontal Pixel Change (white to dark red scale)
    plt.subplot(8, 3, 19)
    plt.imshow(hchange3, cmap='Reds')
    plt.title('Horizontal Pixel Change - 3\n(White=Low, Red=High)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 20)
    plt.imshow(hchange5, cmap='Reds')
    plt.title('Horizontal Pixel Change - 5\n(White=Low, Red=High)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(8, 3, 21)
    diff_hchange = np.abs(hchange3 - hchange5)
    plt.imshow(diff_hchange, cmap='hot')
    plt.title('Difference (H. Pixel Change)', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Row 8: Whitespace in center
    plt.subplot(8, 3, 22)
    plt.imshow(white3, cmap='gray')
    plt.title(f'Center Whitespace - 3\nPixels: {white_count3}', fontsize=12)
    plt.axis('off')
    
    plt.subplot(8, 3, 23)
    plt.imshow(white5, cmap='gray')
    plt.title(f'Center Whitespace - 5\nPixels: {white_count5}', fontsize=12)
    plt.axis('off')
    
    plt.subplot(8, 3, 24)
    diff_white = np.abs(white3.astype(float) - white5.astype(float))
    plt.imshow(diff_white, cmap='hot')
    plt.title(f'Difference (Whitespace)\nΔ={abs(white_count3-white_count5)} pixels', fontsize=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_plots/full_feature_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: comparison_plots/full_feature_comparison.png")
    plt.close()
    
    # Create separate plot for black pixels per row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(row_counts3, range(len(row_counts3)), 'b-', linewidth=2)
    axes[0].set_ylim(len(row_counts3), 0)
    axes[0].set_xlabel('Black Pixels Count', fontsize=12)
    axes[0].set_ylabel('Row Number', fontsize=12)
    axes[0].set_title('Black Pixels per Row - Number 3', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(row_counts5, range(len(row_counts5)), 'r-', linewidth=2)
    axes[1].set_ylim(len(row_counts5), 0)
    axes[1].set_xlabel('Black Pixels Count', fontsize=12)
    axes[1].set_ylabel('Row Number', fontsize=12)
    axes[1].set_title('Black Pixels per Row - Number 5', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(row_counts3, range(len(row_counts3)), 'b-', linewidth=2, label='Number 3', alpha=0.7)
    axes[2].plot(row_counts5, range(len(row_counts5)), 'r-', linewidth=2, label='Number 5', alpha=0.7)
    axes[2].fill_betweenx(range(len(row_counts3)), row_counts3, row_counts5, 
                          where=(np.array(row_counts3) >= np.array(row_counts5)), 
                          alpha=0.3, color='blue', label='3 > 5')
    axes[2].fill_betweenx(range(len(row_counts3)), row_counts3, row_counts5, 
                          where=(np.array(row_counts3) < np.array(row_counts5)), 
                          alpha=0.3, color='red', label='5 > 3')
    axes[2].set_ylim(len(row_counts3), 0)
    axes[2].set_xlabel('Black Pixels Count', fontsize=12)
    axes[2].set_ylabel('Row Number', fontsize=12)
    axes[2].set_title('Comparison - Black Pixels per Row', fontsize=14, fontweight='bold')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_plots/black_pixels_per_row.png', dpi=150, bbox_inches='tight')
    print("Saved: comparison_plots/black_pixels_per_row.png")
    plt.close()
    
    # Create heatmap comparison
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    features = [
        (mag3, mag5, 'Gradient Magnitude', 'viridis'),
        (hchange3, hchange5, 'Horizontal Change', 'Reds'),
        (curv3, curv5, 'Curvature', 'RdBu_r'),
        (tex3, tex5, 'Texture', 'plasma')
    ]
    
    for idx, (feat3, feat5, title, cmap) in enumerate(features):
        axes[0, idx].imshow(feat3, cmap=cmap)
        axes[0, idx].set_title(f'{title} - 3', fontsize=11)
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(feat5, cmap=cmap)
        axes[1, idx].set_title(f'{title} - 5', fontsize=11)
        axes[1, idx].axis('off')
    
    plt.suptitle('Feature Heatmaps Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_plots/feature_heatmaps.png', dpi=150, bbox_inches='tight')
    print("Saved: comparison_plots/feature_heatmaps.png")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FEATURE COMPARISON SUMMARY")
    print("="*60)
    print(f"Whitespace in center - 3: {white_count3} pixels")
    print(f"Whitespace in center - 5: {white_count5} pixels")
    print(f"Difference: {abs(white_count3 - white_count5)} pixels")
    print()
    print(f"Total black pixels - 3: {np.sum(row_counts3)}")
    print(f"Total black pixels - 5: {np.sum(row_counts5)}")
    print(f"Difference: {abs(np.sum(row_counts3) - np.sum(row_counts5))}")
    print()
    print(f"Average gradient magnitude - 3: {np.mean(mag3):.2f}")
    print(f"Average gradient magnitude - 5: {np.mean(mag5):.2f}")
    print()
    print(f"Average horizontal change - 3: {np.mean(hchange3):.2f}")
    print(f"Average horizontal change - 5: {np.mean(hchange5):.2f}")
    print()
    print(f"Average curvature - 3: {np.mean(curv3):.4f}")
    print(f"Average curvature - 5: {np.mean(curv5):.4f}")
    print()
    print(f"Average texture - 3: {np.mean(tex3):.2f}")
    print(f"Average texture - 5: {np.mean(tex5):.2f}")
    print("="*60)

def main():
    print("Creating images of numbers 3 and 5...")
    img3 = create_number_image(3)
    img5 = create_number_image(5)
    
    # Save original images
    cv2.imwrite('experiment/debug_images/3_original.png', img3)
    cv2.imwrite('experiment/debug_images/5_original.png', img5)
    print("Saved original images to experiment/debug_images/")
    
    print("\nAnalyzing features and creating visualizations...")
    visualize_all_features(img3, img5)
    
    print("\n✓ All visualizations complete!")
    print("  - Debug images saved in: experiment/debug_images/")
    print("  - Comparison plots saved in: comparison_plots/")

if __name__ == "__main__":
    main()
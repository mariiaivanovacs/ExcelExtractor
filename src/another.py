#!/usr/bin/env python3
"""
another.py

Advanced feature extraction script for cell image classification.
Generates 300 samples per class (NUMBER, WORD, OTHER) and extracts detailed features
including shape analysis, contour structure, edge density, connected components,
and classification-specific metrics.

Features extracted:
1. Shape and contour structure (curves, irregularities, ascenders/descenders)
2. Character width-to-height ratios and aspect ratio variations
3. Density and frequency of edges using Canny and Sobel filters
4. Connected-component statistics (holes, blobs, stroke count)
5. Fourier descriptors and HOG features
6. Advanced morphological and statistical features

Output: features_second.csv with 300 samples per class
"""

import numpy as np
import cv2
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.feature import hog
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')

# Import from existing module
from try_knn_type import CellImageGenerator, FeatureExtractor

class AdvancedFeatureExtractor:
    """Advanced feature extractor for detailed cell image analysis."""
    
    @staticmethod
    def extract_shape_contour_features(img: np.ndarray) -> dict:
        """Extract shape and contour structure features."""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'contour_count': 0,
                'avg_contour_area': 0,
                'contour_solidity': 0,
                'contour_extent': 0,
                'contour_perimeter_ratio': 0,
                'hull_area_ratio': 0,
                'contour_complexity': 0
            }
        
        # Filter significant contours
        significant_contours = [c for c in contours if cv2.contourArea(c) > 10]
        
        if not significant_contours:
            return {
                'contour_count': 0,
                'contour_solidity': 0,
                'contour_extent': 0,
                'contour_perimeter_ratio': 0,
                'hull_area_ratio': 0,
                'contour_complexity': 0
            }
        
        # Calculate contour features
        areas = [cv2.contourArea(c) for c in significant_contours]
        perimeters = [cv2.arcLength(c, True) for c in significant_contours]
        
        # Solidity (area/convex_hull_area)
        solidities = []
        extents = []
        hull_ratios = []
        
        for contour in significant_contours:
            area = cv2.contourArea(contour)
            if area > 0:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                solidities.append(solidity)
                hull_ratios.append(hull_area / area if area > 0 else 0)
                
                # Extent (area/bounding_rect_area)
                x, y, w, h = cv2.boundingRect(contour)
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                extents.append(extent)
        
        # Contour complexity (perimeter^2 / area)
        complexities = []
        for i, contour in enumerate(significant_contours):
            if i < len(perimeters) and i < len(areas) and areas[i] > 0:
                complexity = (perimeters[i] ** 2) / (4 * np.pi * areas[i])
                complexities.append(complexity)
        
        return {
            'contour_count': len(significant_contours),
            'avg_contour_area': np.mean(areas) if areas else 0,
            'contour_solidity': np.mean(solidities) if solidities else 0,
            'contour_extent': np.mean(extents) if extents else 0,
            'contour_perimeter_ratio': np.mean([p/a if a > 0 else 0 for p, a in zip(perimeters, areas)]),
            'hull_area_ratio': np.mean(hull_ratios) if hull_ratios else 0,
            'contour_complexity': np.mean(complexities) if complexities else 0
        }
    
    @staticmethod
    def extract_aspect_ratio_features(img: np.ndarray) -> dict:
        """Extract character width-to-height ratio features."""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(img_uint8)
        
        if num_labels <= 1:  # Only background
            return {
                'aspect_ratio_mean': 1.0,
                'aspect_ratio_std': 0.0,
                'aspect_ratio_range': 0.0,
                'width_variation': 0.0,
                'height_variation': 0.0,
                'char_count_estimate': 0
            }
        
        aspect_ratios = []
        widths = []
        heights = []
        
        for label_id in range(1, num_labels):  # Skip background (0)
            mask = (labels == label_id).astype(np.uint8)
            coords = cv2.findNonZero(mask)
            
            if coords is not None and len(coords) > 5:  # Minimum size threshold
                x, y, w, h = cv2.boundingRect(coords)
                if h > 0:
                    aspect_ratio = w / h
                    aspect_ratios.append(aspect_ratio)
                    widths.append(w)
                    heights.append(h)
        
        if not aspect_ratios:
            return {
                'aspect_ratio_mean': 1.0,
                'aspect_ratio_std': 0.0,
                'aspect_ratio_range': 0.0,
                'width_variation': 0.0,
                'height_variation': 0.0,
                'char_count_estimate': 0
            }
        
        return {
            'aspect_ratio_mean': np.mean(aspect_ratios),
            'aspect_ratio_std': np.std(aspect_ratios),
            'aspect_ratio_range': np.max(aspect_ratios) - np.min(aspect_ratios),
            'width_variation': np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 0,
            'height_variation': np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 0,
            'char_count_estimate': len(aspect_ratios)
        }
    
    @staticmethod
    def extract_edge_density_features(img: np.ndarray) -> dict:
        """Extract edge density and frequency features using Canny and Sobel."""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Canny edge detection
        edges_canny = cv2.Canny(img_uint8, 50, 150)
        canny_density = np.sum(edges_canny > 0) / edges_canny.size
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Edge statistics
        sobel_mean = np.mean(sobel_magnitude)
        sobel_std = np.std(sobel_magnitude)
        sobel_density = np.sum(sobel_magnitude > sobel_mean) / sobel_magnitude.size
        
        # Directional edge analysis
        sobel_horizontal = np.mean(np.abs(sobel_x))
        sobel_vertical = np.mean(np.abs(sobel_y))
        edge_direction_ratio = sobel_horizontal / (sobel_vertical + 1e-8)
        
        # Edge frequency analysis (FFT-based)
        fft = np.fft.fft2(img)
        fft_magnitude = np.abs(fft)
        high_freq_energy = np.sum(fft_magnitude[fft_magnitude.shape[0]//4:, fft_magnitude.shape[1]//4:])
        total_energy = np.sum(fft_magnitude)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        return {
            'canny_edge_density': canny_density,
            'sobel_edge_density': sobel_density,
            'sobel_mean_magnitude': sobel_mean,
            'sobel_std_magnitude': sobel_std,
            'edge_direction_ratio': edge_direction_ratio,
            'high_freq_ratio': high_freq_ratio
        }
    
    @staticmethod
    def extract_connected_component_features(img: np.ndarray) -> dict:
        """Extract connected component statistics including holes and blobs."""
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Connected components analysis
        num_labels, labels = cv2.connectedComponents(img_uint8)
        
        # Hole detection (using inverted image)
        img_inv = 255 - img_uint8
        num_holes, _ = cv2.connectedComponents(img_inv)
        
        # Region properties using skimage
        binary_img = img_uint8 > 127
        labeled_img = label(binary_img)
        regions = regionprops(labeled_img)
        
        if not regions:
            return {
                'cc_count': 0,
                'hole_count': 0,
                'avg_cc_area': 0,
                'cc_area_std': 0,
                'avg_eccentricity': 0,
                'avg_euler_number': 0,
                'stroke_density': 0
            }
        
        # Extract region properties
        areas = [region.area for region in regions]
        eccentricities = [region.eccentricity for region in regions]
        euler_numbers = [region.euler_number for region in regions]
        
        # Stroke density estimation
        total_area = np.sum(binary_img)
        image_area = binary_img.size
        stroke_density = total_area / image_area
        
        return {
            'cc_count': len(regions),
            'hole_count': max(0, num_holes - 1),  # Subtract background
            'avg_cc_area': np.mean(areas),
            'cc_area_std': np.std(areas),
            'avg_eccentricity': np.mean(eccentricities),
            'avg_euler_number': np.mean(euler_numbers),
            'stroke_density': stroke_density
        }

    @staticmethod
    def extract_hog_features(img: np.ndarray) -> dict:
        """Extract HOG (Histogram of Oriented Gradients) features."""
        img_uint8 = (img * 255).astype(np.uint8)

        # Calculate HOG features
        try:
            hog_features = hog(img_uint8,
                             orientations=9,
                             pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2),
                             block_norm='L2-Hys',
                             visualize=False)

            # Statistical measures of HOG features
            hog_mean = np.mean(hog_features)
            hog_std = np.std(hog_features)
            hog_max = np.max(hog_features)
            hog_energy = np.sum(hog_features**2)

            return {
                'hog_mean': hog_mean,
                'hog_std': hog_std,
                'hog_max': hog_max,
                'hog_energy': hog_energy,
                'hog_feature_count': len(hog_features)
            }
        except:
            return {
                'hog_mean': 0,
                'hog_std': 0,
                'hog_max': 0,
                'hog_energy': 0,
                'hog_feature_count': 0
            }

    @staticmethod
    def extract_fourier_descriptors(img: np.ndarray) -> dict:
        """Extract Fourier descriptors for shape analysis."""
        img_uint8 = (img * 255).astype(np.uint8)

        # Find the largest contour
        contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                'fourier_energy': 0,
                'fourier_symmetry': 0,
                'fourier_complexity': 0
            }

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 10:  # Need sufficient points
            return {
                'fourier_energy': 0,
                'fourier_symmetry': 0,
                'fourier_complexity': 0
            }

        # Convert contour to complex numbers
        contour_points = largest_contour.reshape(-1, 2)
        complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]

        # Calculate Fourier transform
        fourier_coeffs = np.fft.fft(complex_contour)
        fourier_magnitudes = np.abs(fourier_coeffs)

        # Fourier descriptors
        fourier_energy = np.sum(fourier_magnitudes**2)
        fourier_symmetry = fourier_magnitudes[1] / (fourier_magnitudes[0] + 1e-8)
        fourier_complexity = np.std(fourier_magnitudes) / (np.mean(fourier_magnitudes) + 1e-8)

        return {
            'fourier_energy': fourier_energy,
            'fourier_symmetry': fourier_symmetry,
            'fourier_complexity': fourier_complexity
        }

    @staticmethod
    def extract_all_advanced_features(img: np.ndarray) -> dict:
        """Extract all advanced features from an image."""
        features = {}

        # Extract all feature groups
        features.update(AdvancedFeatureExtractor.extract_shape_contour_features(img))
        features.update(AdvancedFeatureExtractor.extract_aspect_ratio_features(img))
        features.update(AdvancedFeatureExtractor.extract_edge_density_features(img))
        features.update(AdvancedFeatureExtractor.extract_connected_component_features(img))
        features.update(AdvancedFeatureExtractor.extract_hog_features(img))
        features.update(AdvancedFeatureExtractor.extract_fourier_descriptors(img))

        return features


def generate_advanced_dataset(n_samples_per_class: int = 300,
                            img_size: tuple = (160,40)) -> pd.DataFrame:
    """
    Generate dataset with 300 samples per class and advanced features.

    Args:
        n_samples_per_class: Number of samples per class (default: 300)
        img_size: Image size (width, height)

    Returns:
        DataFrame with advanced features and labels
    """
    print(f"Generating advanced dataset: {n_samples_per_class} samples per class")
    print(f"Image size: {img_size}")
    print("="*60)

    generator = CellImageGenerator(img_size)
    basic_extractor = FeatureExtractor()
    advanced_extractor = AdvancedFeatureExtractor()

    data = []

    # Generate NUMBER samples
    print("Generating NUMBERS...")
    for i in range(n_samples_per_class):
        img = generator.generate_number()

        # Extract basic features
        basic_features = basic_extractor.extract_features(img)

        # Extract advanced features
        advanced_features = advanced_extractor.extract_all_advanced_features(img)

        # Combine features
        features = {**basic_features, **advanced_features}
        features['label'] = 'NUMBER'
        features['sample_id'] = f'NUM_{i:03d}'

        data.append(features)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples_per_class} NUMBER samples")

    # Generate WORD samples
    print("Generating WORDS...")
    for i in range(n_samples_per_class):
        img = generator.generate_word()

        # Extract basic features
        basic_features = basic_extractor.extract_features(img)

        # Extract advanced features
        advanced_features = advanced_extractor.extract_all_advanced_features(img)

        # Combine features
        features = {**basic_features, **advanced_features}
        features['label'] = 'WORD'
        features['sample_id'] = f'WORD_{i:03d}'

        data.append(features)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples_per_class} WORD samples")

    # Generate OTHER samples
    print("Generating OTHER...")
    for i in range(n_samples_per_class):
        img = generator.generate_other()

        # Extract basic features
        basic_features = basic_extractor.extract_features(img)

        # Extract advanced features
        advanced_features = advanced_extractor.extract_all_advanced_features(img)

        # Combine features
        features = {**basic_features, **advanced_features}
        features['label'] = 'OTHER'
        features['sample_id'] = f'OTHER_{i:03d}'

        data.append(features)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_samples_per_class} OTHER samples")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Reorder columns
    cols = ['sample_id', 'label'] + [c for c in df.columns if c not in ['sample_id', 'label']]
    df = df[cols]

    print("\nAdvanced dataset generated!")
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())

    return df


def main():
    """Main function to generate dataset and save to CSV."""
    print("="*60)
    print("ADVANCED FEATURE EXTRACTION FOR CELL CLASSIFICATION")
    print("="*60)

    # Generate dataset with 300 samples per class
    df = generate_advanced_dataset(n_samples_per_class=300, img_size=(160,40))

    # Save to CSV
    csv_path = 'results/features_second.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved to: {csv_path}")

    # Print feature summary
    print(f"\nFeature summary:")
    print(f"Total features: {len(df.columns) - 2}")  # Exclude sample_id and label
    print(f"Feature names: {[col for col in df.columns if col not in ['sample_id', 'label']]}")

    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)
    print(f"Output file: {csv_path}")
    print(f"Total samples: {len(df)} (300 per class)")
    print(f"Classes: NUMBER, WORD, OTHER")

    return df


if __name__ == '__main__':
    df = main()

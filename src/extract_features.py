import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Import from existing module
from try_knn_type import CellImageGenerator, FeatureExtractor
from another import AdvancedFeatureExtractor


# Your FeatureExtractor class code should be imported or included here
# For this script, I'm assuming it's available in the same file or imported

class FeatureExtractor:
    """Feature extraction from cell images."""
    
    @staticmethod
    def extract_features(img: np.ndarray) -> Dict[str, float]:
        """Extract all features from image."""
        # Convert to uint8 for OpenCV
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Invert for processing (black text on white -> white text on black)
        img_inv = 255 - img_uint8
        
        # Extract all features
        dark_frac = np.sum(img < 0.5) / img.size
        vert_strokes = FeatureExtractor._count_vertical_strokes(img_uint8)
        horiz_strokes = FeatureExtractor._count_horizontal_strokes(img_uint8)
        aspect_ratio = FeatureExtractor._calculate_aspect_ratio(img_inv)
        compactness = FeatureExtractor._calculate_compactness(img_inv)
        num_contours = FeatureExtractor._count_contours(img_uint8)
        cc_count = FeatureExtractor._count_connected_components(img_inv)
        col_peaks = FeatureExtractor._count_column_peaks(img_uint8)
        row_peaks = FeatureExtractor._count_row_peaks(img_uint8)
        vertical_intensity_variance = FeatureExtractor._vertical_intensity_variance(img_uint8)
        gap_depth_index = FeatureExtractor._gap_depth_index(img_uint8)
        intensity_fluctuation_ratio = FeatureExtractor._intensity_fluctuation_ratio(img_uint8)
        column_intensity_entropy = FeatureExtractor._column_intensity_entropy(img_uint8)
        frequency_white, average_peak_width = aspect_ratio_frequency(img_uint8)
        
        return {
            'dark_frac': dark_frac,
            'vert_strokes_count': vert_strokes,
            'horiz_strokes_count': horiz_strokes,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'num_contours': num_contours,
            'cc_count': cc_count,
            'col_peaks': col_peaks,
            'row_peaks': row_peaks,
            'vertical_intensity_variance': vertical_intensity_variance,
            'gap_depth_index': gap_depth_index,
            'intensity_fluctuation_ratio': intensity_fluctuation_ratio,
            'column_intensity_entropy': column_intensity_entropy,
            'frequency_white': frequency_white,
            'average_peak_width': average_peak_width,
        }
    
    @staticmethod
    def _count_vertical_strokes(img: np.ndarray) -> int:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        _, binary = cv2.threshold(vertical, 127, 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(binary)
        return max(0, num_labels - 1)
    
    @staticmethod
    def _count_horizontal_strokes(img: np.ndarray) -> int:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        _, binary = cv2.threshold(horizontal, 127, 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(binary)
        return max(0, num_labels - 1)
    
    @staticmethod
    def _calculate_aspect_ratio(img: np.ndarray) -> float:
        coords = cv2.findNonZero(img)
        if coords is None or len(coords) < 2:
            return 1.0
        x, y, w, h = cv2.boundingRect(coords)
        if h == 0:
            return 1.0
        aspect_ratio = w / h
        aspect_ratio = aspect_ratio / np.max([w, h])
        return aspect_ratio
    
    @staticmethod
    def _calculate_compactness(img: np.ndarray) -> float:
        coords = cv2.findNonZero(img)
        if coords is None or len(coords) < 2:
            return 0.0
        x, y, w, h = cv2.boundingRect(coords)
        bbox_area = w * h
        if bbox_area == 0:
            return 0.0
        content_area = np.sum(img > 0)
        return content_area / bbox_area
    
    @staticmethod
    def _count_contours(img: np.ndarray) -> int:
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours if cv2.contourArea(c) > 5]
        return len(significant_contours)
    
    @staticmethod
    def _count_connected_components(img: np.ndarray) -> int:
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        num_labels, _ = cv2.connectedComponents(binary)
        return max(0, num_labels - 1)
    
    @staticmethod
    def _count_column_peaks(img: np.ndarray) -> int:
        projection = np.sum(img, axis=0)
        if np.max(projection) < 10:
            return 0
        if np.max(projection) > 0:
            projection = projection / np.max(projection)
        peaks, _ = find_peaks(projection, height=0.2, distance=3)
        return len(peaks)
    
    @staticmethod
    def _count_row_peaks(img: np.ndarray) -> int:
        projection = np.sum(img, axis=1)
        if np.max(projection) < 10:
            return 0
        if np.max(projection) > 0:
            projection = projection / np.max(projection)
        peaks, _ = find_peaks(projection, height=0.2, distance=2)
        return len(peaks)
    
    @staticmethod
    def _column_profile(img: np.ndarray) -> np.ndarray:
        gray = img
        col_mean = gray.mean(axis=0)
        col_profile = 1.0 - col_mean
        return col_profile.astype(np.float32)

    @staticmethod
    def _vertical_intensity_variance(img: np.ndarray) -> float:
        col = FeatureExtractor._column_profile(img)
        if col.size == 0:
            return 0.0
        return float(np.var(col))

    @staticmethod
    def _gap_depth_index(img: np.ndarray) -> float:
        col = FeatureExtractor._column_profile(img)
        n = col.size
        if n < 3:
            return 0.0
        minv = float(col.min())
        maxv = float(col.max())
        rng = maxv - minv
        if rng <= 1e-8:
            return 0.0
        norm = (col - minv) / rng
        left = norm[:-2]
        center = norm[1:-1]
        right = norm[2:]
        maxima_idx = np.where((center > left) & (center >= right))[0] + 1
        minima_idx = np.where((center < left) & (center <= right))[0] + 1
        if maxima_idx.size < 2 or minima_idx.size == 0:
            return float(1.0 if rng > 0 else 0.0) * float((maxv - minv) / (maxv + 1e-9))
        depths = []
        for valley in minima_idx:
            left_peaks = maxima_idx[maxima_idx < valley]
            right_peaks = maxima_idx[maxima_idx > valley]
            if left_peaks.size == 0 or right_peaks.size == 0:
                continue
            left_peak = left_peaks[-1]
            right_peak = right_peaks[0]
            peak_height = min(norm[left_peak], norm[right_peak])
            valley_val = norm[valley]
            depth = peak_height - valley_val
            if depth > 0:
                depths.append(depth)
        if len(depths) == 0:
            return float((norm.max() - norm.min()))
        avg_depth = float(np.mean(depths))
        return float(np.clip(avg_depth, 0.0, 1.0))

    @staticmethod
    def _intensity_fluctuation_ratio(img: np.ndarray) -> float:
        col = FeatureExtractor._column_profile(img)
        if col.size < 2:
            return 0.0
        diffs = np.abs(np.diff(col))
        mean_abs_diff = float(diffs.mean())
        mean_col = float(np.abs(col).mean())
        eps = 1e-9
        ratio = mean_abs_diff / (mean_col + eps)
        return float(ratio)

    @staticmethod
    def _column_intensity_entropy(img: np.ndarray, bins: int = 32) -> float:
        col = FeatureExtractor._column_profile(img)
        if col.size == 0:
            return 0.0
        counts, _ = np.histogram(col, bins=bins, range=(0.0, 1.0))
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts.astype(np.float64) / float(total)
        p_nonzero = p[p > 0.0]
        entropy = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
        max_ent = np.log2(bins)
        if max_ent <= 0:
            return 0.0
        return float(entropy / max_ent)


def aspect_ratio_frequency(img):
    """Calculate aspect ratio frequency features."""
    # Calculate column intensities
    img_float = img.astype(np.float32) / 255.0
    inverted = 1.0 - img_float
    column_intensities = np.mean(inverted, axis=0)
    
    threshold = 0.05
    col = np.asarray(column_intensities).astype(float).ravel()
    n = col.size
    if n == 0:
        return 0.0, 0.0
    
    if col.max() > 1.0:
        col = col / float(col.max())
    
    states = col > threshold
    false_count = np.sum(states == False)
    ratio = false_count / img.shape[1]
    
    run_start = 0
    in_character = False
    peaks = []
    
    for i in range(len(states)):
        if states[i] == True:
            if not in_character:
                run_start = i
                in_character = True
        else:
            if in_character:
                run_end = i
                run_width = run_end - run_start
                if run_width >= 2:
                    peaks.append((run_end - run_start + 2))
                in_character = False
    
    amount_of_peaks = len(peaks)
    if amount_of_peaks > 0:
        average_peak_width = sum(peaks) / amount_of_peaks
    else:
        average_peak_width = 0
    
    average_peak_width = average_peak_width / img.shape[1]
    return ratio, average_peak_width


def process_images_to_csv(input_folder: str, output_csv: str):
    """
    Process all PNG images in a folder and extract features to CSV.
    
    Args:
        input_folder: Path to folder containing PNG images
        output_csv: Path to output CSV file
    """
    
    advanced_extractor = AdvancedFeatureExtractor()
    basic_extractor = FeatureExtractor()


    # Get all PNG files
    image_files = sorted(Path(input_folder).glob("*.png"))
    
    if not image_files:
        print(f"No PNG files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} PNG images")
    
    # List to store results
    results = []
    
    # Process each image
    for idx, img_path in enumerate(image_files, 1):
        try:
            # Load image as grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Could not load {img_path.name}")
                continue
            
            # Normalize to [0, 1]
            img_normalized = img.astype(np.float32) / 255.0
            
            basic_features = basic_extractor.extract_features(img_normalized)
            
            advanced_features = advanced_extractor.extract_all_advanced_features(img_normalized)

            # Extract features
            features = {**basic_features, **advanced_features}
            
            # Add sample_id (filename without extension)
            features['sample_id'] = img_path.stem
            
            results.append(features)
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Create DataFrame with specified column order
    columns = [
        'sample_id', 'dark_frac', 'vert_strokes_count', 'horiz_strokes_count',
        'aspect_ratio', 'compactness', 'num_contours', 'cc_count', 
        'col_peaks', 'row_peaks', 'vertical_intensity_variance',
        'gap_depth_index', 'intensity_fluctuation_ratio', 
        'column_intensity_entropy', 'frequency_white', 'average_peak_width'
    ]
    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['sample_id'] + [c for c in df.columns if c not in ['sample_id', 'label']]
    df = df[cols]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully processed {len(results)} images")
    print(f"Results saved to: {output_csv}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "words_production"
    OUTPUT_CSV = "results/features_output.csv"
    
    # Run processing
    process_images_to_csv(INPUT_FOLDER, OUTPUT_CSV)
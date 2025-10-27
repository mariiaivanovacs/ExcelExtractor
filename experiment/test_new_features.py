#!/usr/bin/env python3
"""
Test script to verify the new advanced features work correctly.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# Add the experiment directory to path to import fight.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fight import extract_features

def create_test_digit(digit, size=32):
    """Create a simple test digit image."""
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Try to use a simple font, fallback to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Get text size and center it
    try:
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        # Fallback for older PIL versions
        text_width, text_height = draw.textsize(str(digit), font=font)
    
    position = ((size - text_width) // 2, (size - text_height) // 2)
    draw.text(position, str(digit), fill=255, font=font)
    
    return np.array(img)

def test_new_features():
    """Test the new advanced features."""
    print("=== Testing New Advanced Features ===\n")
    
    # Test with different digits
    test_digits = [0, 1, 8, 9]  # Different shapes for testing
    
    for digit in test_digits:
        print(f"Testing digit: {digit}")
        
        # Create test image
        test_image = create_test_digit(digit)
        
        # Extract features
        try:
            features = extract_features(test_image)
            
            # Print the new features
            print(f"  Projection correlation: {features.get('projection_correlation', 'N/A'):.4f}")
            print(f"  Energy ratio (upper/lower): {features.get('energy_ratio', 'N/A'):.4f}")
            print(f"  Quadrant energy ratios:")
            print(f"    Q1 (top-left): {features.get('q1_energy_ratio', 'N/A'):.4f}")
            print(f"    Q2 (top-right): {features.get('q2_energy_ratio', 'N/A'):.4f}")
            print(f"    Q3 (bottom-left): {features.get('q3_energy_ratio', 'N/A'):.4f}")
            print(f"    Q4 (bottom-right): {features.get('q4_energy_ratio', 'N/A'):.4f}")
            print(f"  Quadrant energy variance: {features.get('quadrant_energy_variance', 'N/A'):.4f}")
            print(f"  Row/Column variance ratio: {features.get('row_variance_ratio', 'N/A'):.4f}")
            print(f"  Row entropy: {features.get('row_entropy', 'N/A'):.4f}")
            print(f"  Column entropy: {features.get('col_entropy', 'N/A'):.4f}")
            print(f"  Projection entropy ratio: {features.get('projection_entropy_ratio', 'N/A'):.4f}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    print("✅ Feature extraction test completed!")

def test_feature_consistency():
    """Test that features are consistent and reasonable."""
    print("=== Testing Feature Consistency ===\n")
    
    # Create a simple test image
    test_image = create_test_digit(8)  # Digit 8 has interesting properties
    
    try:
        features = extract_features(test_image)
        
        # Check that quadrant ratios sum to 1
        quad_sum = (features.get('q1_energy_ratio', 0) + 
                   features.get('q2_energy_ratio', 0) + 
                   features.get('q3_energy_ratio', 0) + 
                   features.get('q4_energy_ratio', 0))
        
        print(f"Quadrant energy ratios sum: {quad_sum:.4f} (should be ~1.0)")
        
        if abs(quad_sum - 1.0) < 0.01:
            print("✅ Quadrant ratios sum correctly")
        else:
            print("⚠️  Quadrant ratios don't sum to 1.0")
        
        # Check that correlation is between -1 and 1
        corr = features.get('projection_correlation', 0)
        if -1.0 <= corr <= 1.0:
            print(f"✅ Projection correlation in valid range: {corr:.4f}")
        else:
            print(f"⚠️  Projection correlation out of range: {corr:.4f}")
        
        # Check that entropies are positive
        row_entropy = features.get('row_entropy', 0)
        col_entropy = features.get('col_entropy', 0)
        
        if row_entropy >= 0 and col_entropy >= 0:
            print(f"✅ Entropies are positive: row={row_entropy:.4f}, col={col_entropy:.4f}")
        else:
            print(f"⚠️  Negative entropy detected: row={row_entropy:.4f}, col={col_entropy:.4f}")
        
        print("\n✅ Consistency tests completed!")
        
    except Exception as e:
        print(f"❌ Error in consistency test: {e}")
        import traceback
        traceback.print_exc()

def compare_digits():
    """Compare features between different digits to see discrimination."""
    print("=== Comparing Features Between Digits ===\n")
    
    digits = [0, 1, 8]
    features_list = []
    
    for digit in digits:
        test_image = create_test_digit(digit)
        try:
            features = extract_features(test_image)
            features_list.append((digit, features))
        except Exception as e:
            print(f"Error processing digit {digit}: {e}")
            continue
    
    if len(features_list) >= 2:
        print("Feature comparison (showing differences that might help discrimination):")
        print()
        
        feature_names = [
            'projection_correlation', 'energy_ratio', 'quadrant_energy_variance',
            'row_variance_ratio', 'projection_entropy_ratio'
        ]
        
        for feature_name in feature_names:
            print(f"{feature_name}:")
            for digit, features in features_list:
                value = features.get(feature_name, 0)
                print(f"  Digit {digit}: {value:.4f}")
            print()
    
    print("✅ Feature comparison completed!")

if __name__ == "__main__":
    print("Testing New Advanced Features for Digit Recognition\n")
    
    # Create debug output directory
    os.makedirs("experiment/debug_out", exist_ok=True)
    
    try:
        # Test basic functionality
        test_new_features()
        
        # Test consistency
        test_feature_consistency()
        
        # Compare between digits
        compare_digits()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

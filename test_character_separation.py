#!/usr/bin/env python3
"""
Test script for character separation functionality.
"""

import cv2
import os
from src.seg_cells import process_single_cell_main, separate_characters_advanced

def test_character_separation():
    """Test the character separation on different cell images."""
    
    # Test images with different characteristics
    test_cases = [
        {
            "path": "cells_production/cell_r1_c1.png",
            "description": "Wide cell with multiple characters"
        },
        {
            "path": "cells_production/cell_r20_c16.png", 
            "description": "Cell with dense text"
        },
        {
            "path": "cells_production/cell_r10_c5.png",
            "description": "Small cell with few characters"
        }
    ]
    
    print("Testing Character Separation Algorithm")
    print("="*50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"File: {test_case['path']}")
        print("-" * 40)
        
        if not os.path.exists(test_case['path']):
            print(f"❌ File not found: {test_case['path']}")
            continue
            
        try:
            # Load image
            img = cv2.imread(test_case['path'])
            if img is None:
                print(f"❌ Could not load image: {test_case['path']}")
                continue
                
            # Test character separation
            characters = separate_characters_advanced(img, debug=False)
            
            print(f"✅ Successfully separated {len(characters)} characters")
            for j, char in enumerate(characters):
                print(f"   Character {j}: {char.shape[1]}x{char.shape[0]} pixels")
                
        except Exception as e:
            print(f"❌ Error processing {test_case['path']}: {e}")
    
    # Check results folder
    results_dir = "results"
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        print(f"\n📁 Results folder contains {len(files)} character images:")
        for file in sorted(files):
            print(f"   {file}")
    else:
        print(f"\n❌ Results folder '{results_dir}' not found")

if __name__ == "__main__":
    test_character_separation()

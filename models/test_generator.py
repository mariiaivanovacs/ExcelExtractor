#!/usr/bin/env python3
"""
Test script to verify the memory-efficient generator works correctly.
This script tests the generator without training a full model.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
import cv2
import random
import string
import gc

def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    """Apply distortion to simulate real-world conditions."""
    max_shift = 3
    h, w = np_img.shape[:2]

    # Step 1: Randomly scale down & up (pixelation)
    scale = random.uniform(0.8, 0.83)
    small = cv2.resize(np_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Step 2: Apply small random affine transform
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    up = cv2.warpAffine(up, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Step 3: Lightly "fade" bright regions
    up = up.astype(np.float32)
    factor = random.uniform(0.2, 0.3)
    mask = up > 155
    up[mask] = up[mask] - (up[mask] - 155) * factor
    up = np.clip(up, 0, 255).astype(np.uint8)

    # Step 4: Optional small blur
    up = cv2.GaussianBlur(up, (3, 3), sigmaX=random.uniform(0.5, 1.0))

    return up

def get_character_set():
    """Get a small character set for testing."""
    characters = []
    
    # Just use digits and a few letters for testing
    characters.extend([str(i) for i in range(10)])  # 0-9
    characters.extend(['A', 'B', 'C', 'D', 'E'])  # A few letters
    
    return characters

def setup_fonts(font_size=12):
    """Setup available fonts for character rendering."""
    fonts = []
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Microsoft/Arial.ttf", font_size)
        fonts.append(font)
    except:
        pass
    
    # Fallback to default font if no fonts loaded
    if not fonts:
        fonts = [ImageFont.load_default()]
    
    return fonts

def generate_single_character_image(char, char_idx, fonts, image_size=32):
    """Generate a single character image with distortion."""
    font = random.choice(fonts)

    # Create a blank grayscale image (black background)
    img = Image.new('L', (image_size, image_size), color=0)
    draw = ImageDraw.Draw(img)

    # Compute centered text position
    try:
        text_width, text_height = draw.textbbox((0, 0), char, font=font)[2:4]
    except:
        # Fallback for older PIL versions
        text_width, text_height = draw.textsize(char, font=font)

    position = ((image_size - text_width) // 2, (image_size - text_height) // 2)

    # Draw white text
    draw.text(position, char, fill=255, font=font)

    # Convert to NumPy array
    np_img = np.array(img)

    # Apply random distortion to simulate real-world conditions
    distorted = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)

    return distorted, char_idx

def character_generator(characters, char_to_label, fonts, num_variants=50, image_size=32, max_batch_size=100):
    """
    Generator that yields batches of character images without storing all in memory.
    """
    batch_images = []
    batch_labels = []
    
    for char_idx, char in enumerate(characters):
        for i in range(num_variants):
            # Generate single image
            image, label = generate_single_character_image(char, char_idx, fonts, image_size)
            
            batch_images.append(image)
            batch_labels.append(label)
            
            # Yield batch when it reaches the specified size
            if len(batch_images) >= max_batch_size:
                yield np.array(batch_images, dtype=np.uint8), np.array(batch_labels, dtype=np.int32)
                batch_images = []
                batch_labels = []
        
        print(f"Processed character '{char}' ({char_idx + 1}/{len(characters)})")
    
    # Yield remaining images if any
    if batch_images:
        yield np.array(batch_images, dtype=np.uint8), np.array(batch_labels, dtype=np.int32)

def get_memory_usage():
    """Get current memory usage in MB (simplified version)."""
    # Simple memory tracking - just return a placeholder
    return 0.0

def test_generator():
    """Test the memory-efficient generator."""
    print("=== Testing Memory-Efficient Generator ===\n")
    
    # Setup
    characters = get_character_set()
    char_to_label = {char: idx for idx, char in enumerate(characters)}
    fonts = setup_fonts()
    
    # Configuration
    num_variants = 200  # images per character
    image_size = 32
    max_batch_size = 1000  # Memory limit
    
    print(f"Character set: {characters}")
    print(f"Total characters: {len(characters)}")
    print(f"Variants per character: {num_variants}")
    print(f"Total images to generate: {len(characters) * num_variants}")
    print(f"Max batch size (memory limit): {max_batch_size}")
    print(f"Image size: {image_size}x{image_size}")
    
    # Test the generator
    total_images_processed = 0
    batch_count = 0

    print("\nStarting generator test...")

    for batch_images, batch_labels in character_generator(
        characters, char_to_label, fonts, num_variants, image_size, max_batch_size
    ):
        batch_count += 1
        total_images_processed += len(batch_images)

        print(f"Batch {batch_count}: {len(batch_images)} images, "
              f"Total processed: {total_images_processed}")

        # Verify batch properties
        assert batch_images.shape[0] == len(batch_labels), "Batch size mismatch"
        assert batch_images.dtype == np.uint8, "Wrong image dtype"
        assert batch_labels.dtype == np.int32, "Wrong label dtype"
        assert batch_images.shape[1:] == (image_size, image_size), "Wrong image shape"

        # Verify we never exceed max_batch_size
        assert len(batch_images) <= max_batch_size, f"Batch size {len(batch_images)} exceeds limit {max_batch_size}"

        # Force garbage collection to test memory cleanup
        del batch_images, batch_labels
        gc.collect()
    
    print(f"\n=== Test Results ===")
    print(f"✓ Total images processed: {total_images_processed}")
    print(f"✓ Total batches: {batch_count}")
    print(f"✓ Expected total images: {len(characters) * num_variants}")

    # Verify we processed all expected images
    expected_total = len(characters) * num_variants
    assert total_images_processed == expected_total, f"Expected {expected_total}, got {total_images_processed}"

    print(f"\n✅ Generator test completed successfully!")
    print(f"The generator processes images in batches of max {max_batch_size}")
    print(f"without storing all {total_images_processed} images in memory at once.")
    print(f"✓ Memory limit enforced: No batch exceeded {max_batch_size} images")

if __name__ == "__main__":
    test_generator()

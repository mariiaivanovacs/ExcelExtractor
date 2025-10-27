#!/usr/bin/env python3
"""
Complete usage example showing how to:
1. Generate synthetic character images
2. Apply downsampling distortion to simulate real-world conditions
3. Train a CNN model
4. Use the trained model for prediction

This demonstrates the complete pipeline from data generation to model inference.
"""

import numpy as np
import tensorflow as tf
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt

def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    """
    Simulate low-resolution, pixelated distortion for white text on black background.
    This mimics the quality degradation that happens in real-world cell images.
    
    Args:
        np_img: Input grayscale image (white text on black background)
        min_scale, max_scale: Downsampling scale range
    
    Returns:
        np.ndarray: Distorted grayscale image
    """
    max_shift = 3
    h, w = np_img.shape[:2]

    # Step 1: Randomly scale down & up (pixelation effect)
    scale = random.uniform(0.8, 0.83)
    small = cv2.resize(np_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Step 2: Apply small random affine transform (shift/jitter)
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    up = cv2.warpAffine(up, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Step 3: Lightly "fade" bright regions (text degradation)
    up = up.astype(np.float32)
    factor = random.uniform(0.2, 0.3)
    mask = up > 155
    up[mask] = up[mask] - (up[mask] - 155) * factor
    up = np.clip(up, 0, 255).astype(np.uint8)

    # Step 4: Optional small blur to soften edges
    up = cv2.GaussianBlur(up, (3, 3), sigmaX=random.uniform(0.5, 1.0))

    return up

def generate_character_image(character, image_size=32, font_size=12, apply_distortion=True):
    """
    Generate a synthetic image of a character.
    
    Args:
        character: Character to render
        image_size: Size of output image (square)
        font_size: Font size for rendering
        apply_distortion: Whether to apply realistic distortions
    
    Returns:
        np.ndarray: Generated image
    """
    # Create a blank grayscale image (black background)
    img = Image.new('L', (image_size, image_size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Try to load a system font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Microsoft/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Compute centered text position
    try:
        text_width, text_height = draw.textbbox((0, 0), character, font=font)[2:4]
    except:
        # Fallback for older PIL versions
        text_width, text_height = draw.textsize(character, font=font)
    
    position = ((image_size - text_width) // 2, (image_size - text_height) // 2)

    # Draw white text on black background
    draw.text(position, character, fill=255, font=font)

    # Convert to NumPy array
    np_img = np.array(img)

    # Apply distortion to simulate real-world conditions
    if apply_distortion:
        np_img = downsample_then_upsample(np_img)
    
    return np_img

def predict_character_from_image(model, image, char_to_label, label_to_char):
    """
    Predict character from an image using the trained model.
    
    Args:
        model: Trained TensorFlow model
        image: Input image (numpy array)
        char_to_label: Character to label mapping
        label_to_char: Label to character mapping
    
    Returns:
        tuple: (predicted_character, confidence, all_probabilities)
    """
    # Preprocess image for model input
    if image.shape != (32, 32):
        image = cv2.resize(image, (32, 32))
    
    img_processed = image.astype("float32") / 255.0
    if img_processed.ndim == 2:
        img_processed = img_processed[..., np.newaxis]  # Add channel dimension
    img_processed = np.expand_dims(img_processed, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_processed, verbose=0)
    predicted_label = np.argmax(predictions[0])
    confidence = predictions[0][predicted_label]
    
    # Convert label back to character
    predicted_char = label_to_char[str(predicted_label)]
    
    return predicted_char, confidence, predictions[0]

def demonstrate_pipeline():
    """
    Demonstrate the complete pipeline from image generation to prediction.
    """
    print("=== Character Recognition Pipeline Demo ===\n")
    
    # Step 1: Load trained model (if available)
    try:
        model = tf.keras.models.load_model("models/digit_cnn.h5")
        with open("models/digit_mappings.json", "r") as f:
            mappings = json.load(f)
        char_to_label = mappings['char_to_label']
        label_to_char = mappings['label_to_char']
        print("✓ Loaded trained digit model")
    except:
        print("✗ No trained model found. Please run train_simple_digits.py first")
        return
    
    # Step 2: Generate test images
    test_characters = ['0', '1', '2', '3', '4', '5']
    
    print("\nStep 1: Generating synthetic character images...")
    
    for char in test_characters:
        # Generate clean image
        clean_img = generate_character_image(char, apply_distortion=False)
        
        # Generate distorted image (simulating real-world conditions)
        distorted_img = generate_character_image(char, apply_distortion=True)
        
        # Save images for inspection
        cv2.imwrite(f"models/demo_clean_{char}.png", clean_img)
        cv2.imwrite(f"models/demo_distorted_{char}.png", distorted_img)
        
        print(f"  Generated images for character '{char}'")
    
    print("\nStep 2: Testing model predictions...")
    
    # Step 3: Test predictions
    correct_clean = 0
    correct_distorted = 0
    total_tests = len(test_characters)
    
    for char in test_characters:
        # Test with clean image
        clean_img = generate_character_image(char, apply_distortion=False)
        pred_clean, conf_clean, _ = predict_character_from_image(
            model, clean_img, char_to_label, label_to_char
        )
        
        # Test with distorted image
        distorted_img = generate_character_image(char, apply_distortion=True)
        pred_distorted, conf_distorted, _ = predict_character_from_image(
            model, distorted_img, char_to_label, label_to_char
        )
        
        # Check accuracy
        if pred_clean == char:
            correct_clean += 1
        if pred_distorted == char:
            correct_distorted += 1
        
        print(f"  Character '{char}':")
        print(f"    Clean:     Predicted '{pred_clean}' (confidence: {conf_clean:.3f})")
        print(f"    Distorted: Predicted '{pred_distorted}' (confidence: {conf_distorted:.3f})")
    
    # Step 4: Show results
    clean_accuracy = correct_clean / total_tests
    distorted_accuracy = correct_distorted / total_tests
    
    print(f"\nStep 3: Results Summary")
    print(f"  Clean images accuracy:     {clean_accuracy:.1%} ({correct_clean}/{total_tests})")
    print(f"  Distorted images accuracy: {distorted_accuracy:.1%} ({correct_distorted}/{total_tests})")
    
    print(f"\nDemo images saved to models/demo_*.png")
    
    # Step 5: Show how to use with real images
    print(f"\nStep 4: Usage with real images")
    print(f"To use this model with real cell images:")
    print(f"1. Load your image: img = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)")
    print(f"2. Resize to 32x32: img = cv2.resize(img, (32, 32))")
    print(f"3. Predict: char, conf, probs = predict_character_from_image(model, img, char_to_label, label_to_char)")
    print(f"4. Use result: print(f'Predicted: {{char}} (confidence: {{conf:.3f}})')")

def create_visualization():
    """Create a visualization showing the effect of distortion."""
    print("\nCreating distortion visualization...")
    
    # Generate examples
    char = '5'
    clean_img = generate_character_image(char, apply_distortion=False)
    distorted_imgs = [generate_character_image(char, apply_distortion=True) for _ in range(4)]
    
    # Create visualization
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    # Show clean image
    axes[0].imshow(clean_img, cmap='gray')
    axes[0].set_title('Clean')
    axes[0].axis('off')
    
    # Show distorted variants
    for i, img in enumerate(distorted_imgs):
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f'Distorted {i+1}')
        axes[i+1].axis('off')
    
    plt.suptitle(f"Character '{char}': Clean vs Distorted Variants")
    plt.tight_layout()
    plt.savefig('models/distortion_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to models/distortion_comparison.png")

if __name__ == "__main__":
    # Run the complete demonstration
    demonstrate_pipeline()
    
    # Create visualization
    create_visualization()
    
    print("\n=== Pipeline Demo Complete ===")
    print("This example shows how to:")
    print("1. Generate synthetic character images")
    print("2. Apply realistic distortions using downsample_then_upsample()")
    print("3. Use a trained CNN model for character prediction")
    print("4. Evaluate model performance on clean vs distorted images")
    print("\nThe downsample_then_upsample() function is key for bridging")
    print("the gap between synthetic training data and real-world image quality.")

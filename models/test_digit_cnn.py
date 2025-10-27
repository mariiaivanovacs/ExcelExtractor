#!/usr/bin/env python3
"""
Test script for the trained digit CNN classifier.
"""

import numpy as np
import tensorflow as tf
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
import random

def load_model_and_mappings():
    """Load the trained model and character mappings."""
    try:
        model = tf.keras.models.load_model("models/digit_cnn.h5")
        
        with open("models/digit_mappings.json", "r", encoding='utf-8') as f:
            mappings = json.load(f)
        
        return model, mappings
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    """Apply the same distortion as used in training."""
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

def create_test_image(character, image_size=32, font_size=12):
    """Create a test image of a character."""
    # Create a blank grayscale image (black background)
    img = Image.new('L', (image_size, image_size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Microsoft/Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Compute centered text position
    try:
        text_width, text_height = draw.textbbox((0, 0), character, font=font)[2:4]
    except:
        text_width, text_height = draw.textsize(character, font=font)
    
    position = ((image_size - text_width) // 2, (image_size - text_height) // 2)

    # Draw white text
    draw.text(position, character, fill=255, font=font)

    # Convert to NumPy array
    np_img = np.array(img)

    # Apply distortion
    distorted = downsample_then_upsample(np_img)
    
    return distorted

def predict_character(model, image, mappings):
    """Predict character from image."""
    # Preprocess image
    img_processed = image.astype("float32") / 255.0
    if img_processed.ndim == 2:
        img_processed = img_processed[..., np.newaxis]  # Add channel dimension
    img_processed = np.expand_dims(img_processed, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(img_processed, verbose=0)
    predicted_label = np.argmax(predictions[0])
    confidence = predictions[0][predicted_label]
    
    # Convert label to character
    predicted_char = mappings['label_to_char'][str(predicted_label)]
    
    return predicted_char, confidence, predictions[0]

def test_model():
    """Test the trained model with some examples."""
    print("Loading digit model and mappings...")
    model, mappings = load_model_and_mappings()
    
    if model is None:
        print("Failed to load model. Make sure to train it first by running train_simple_digits.py")
        return
    
    print(f"Model loaded successfully!")
    print(f"Number of classes: {mappings['num_classes']}")
    print(f"Available characters: {list(mappings['char_to_label'].keys())}")
    
    # Test with all digits
    test_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    print("\nTesting digit predictions:")
    print("-" * 60)
    
    correct_predictions = 0
    total_predictions = 0
    
    for char in test_chars:
        # Test multiple times for each digit
        for test_num in range(3):
            # Create test image
            test_img = create_test_image(char)
            
            # Predict
            pred_char, confidence, all_probs = predict_character(model, test_img, mappings)
            
            # Check if correct
            is_correct = pred_char == char
            correct_predictions += is_correct
            total_predictions += 1
            
            # Show top 3 predictions
            top_3_indices = np.argsort(all_probs)[-3:][::-1]
            top_3_chars = [mappings['label_to_char'][str(i)] for i in top_3_indices]
            top_3_probs = [all_probs[i] for i in top_3_indices]
            
            status = "✓" if is_correct else "✗"
            print(f"{status} True: '{char}' | Predicted: '{pred_char}' | Confidence: {confidence:.3f}")
            print(f"    Top 3: {[(c, f'{p:.3f}') for c, p in zip(top_3_chars, top_3_probs)]}")
            
            # Save test image for inspection
            cv2.imwrite(f"models/test_digit_{char}_{test_num}.png", test_img)
    
    accuracy = correct_predictions / total_predictions
    print(f"\nOverall accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    print(f"Test images saved to models/test_digit_*.png")

def test_with_real_image(image_path):
    """Test the model with a real image file."""
    print(f"\nTesting with real image: {image_path}")
    
    model, mappings = load_model_and_mappings()
    if model is None:
        return
    
    try:
        # Load and preprocess the image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {image_path}")
            return
        
        # Resize to model input size
        img_resized = cv2.resize(img, (32, 32))
        
        # Predict
        pred_char, confidence, all_probs = predict_character(model, img_resized, mappings)
        
        # Show results
        top_3_indices = np.argsort(all_probs)[-3:][::-1]
        top_3_chars = [mappings['label_to_char'][str(i)] for i in top_3_indices]
        top_3_probs = [all_probs[i] for i in top_3_indices]
        
        print(f"Predicted: '{pred_char}' | Confidence: {confidence:.3f}")
        print(f"Top 3: {[(c, f'{p:.3f}') for c, p in zip(top_3_chars, top_3_probs)]}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    test_model()
    
    # Uncomment to test with a real image file
    # test_with_real_image("path/to/your/image.png")

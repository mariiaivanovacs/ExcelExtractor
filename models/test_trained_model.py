#!/usr/bin/env python3
"""
Test script to verify the trained character recognition model works correctly.
"""

import numpy as np
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
import cv2
import random
import tensorflow as tf

def load_model_and_mappings():
    """Load the trained model and character mappings."""
    print("Loading trained model and mappings...")
    
    # Load model
    model = tf.keras.models.load_model('models/char_cnn_trained.h5')
    print(f"✓ Model loaded: {model.input_shape} -> {model.output_shape}")
    
    # Load character mappings
    with open('models/characters_mappings.json', 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    char_to_label = mappings['char_to_label']
    label_to_char = {int(k): v for k, v in mappings['label_to_char'].items()}
    
    print(f"✓ Character mappings loaded: {len(char_to_label)} characters")
    print(f"Sample characters: {list(char_to_label.keys())[:20]}...")
    
    return model, char_to_label, label_to_char

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

def generate_test_character(char, fonts, image_size=32):
    """Generate a test character image."""
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

    # Apply slight distortion
    if random.random() > 0.5:
        # Small blur
        np_img = cv2.GaussianBlur(np_img, (3, 3), sigmaX=random.uniform(0.3, 0.8))
    
    if random.random() > 0.5:
        # Small shift
        max_shift = 2
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        np_img = cv2.warpAffine(np_img, M, (image_size, image_size), borderMode=cv2.BORDER_REPLICATE)

    return np_img

def test_model_predictions():
    """Test the model with various characters."""
    print("\n=== Testing Model Predictions ===")
    
    # Load model and mappings
    model, char_to_label, label_to_char = load_model_and_mappings()
    fonts = setup_fonts()
    
    # Test characters
    test_chars = ['0', '1', '5', '9', 'A', 'B', 'Z', 'a', 'b', 'z', 
                  'А', 'Б', 'Я', 'а', 'б', 'я', '.', ',', '%', 'руб']
    
    correct_predictions = 0
    total_tests = 0
    
    print(f"\nTesting {len(test_chars)} different characters...")
    
    for char in test_chars:
        if char not in char_to_label:
            print(f"⚠️  Character '{char}' not in training set, skipping...")
            continue
            
        # Generate multiple test images for this character
        char_correct = 0
        char_total = 5  # Test 5 variants of each character
        
        for i in range(char_total):
            # Generate test image
            test_image = generate_test_character(char, fonts)
            
            # Prepare for prediction
            test_input = test_image.reshape(1, 32, 32, 1).astype(np.float32) / 255.0
            
            # Make prediction
            predictions = model.predict(test_input, verbose=0)
            predicted_label = np.argmax(predictions[0])
            predicted_char = label_to_char.get(predicted_label, '?')
            confidence = predictions[0][predicted_label]
            
            # Check if correct
            is_correct = predicted_char == char
            if is_correct:
                char_correct += 1
                correct_predictions += 1
            
            total_tests += 1
            
            # Print result for first test of each character
            if i == 0:
                status = "✓" if is_correct else "✗"
                print(f"{status} '{char}' -> '{predicted_char}' (confidence: {confidence:.3f})")
        
        # Print character summary
        char_accuracy = char_correct / char_total
        print(f"  Character '{char}' accuracy: {char_correct}/{char_total} ({char_accuracy:.1%})")
    
    # Print overall results
    overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct_predictions}/{total_tests} ({overall_accuracy:.1%})")
    print(f"Characters tested: {len([c for c in test_chars if c in char_to_label])}")
    
    if overall_accuracy >= 0.7:
        print("✅ Model performance looks good!")
    elif overall_accuracy >= 0.5:
        print("⚠️  Model performance is moderate - may need more training")
    else:
        print("❌ Model performance is poor - needs more training or debugging")
    
    return overall_accuracy

def test_batch_prediction():
    """Test batch prediction with multiple characters."""
    print("\n=== Testing Batch Prediction ===")
    
    # Load model and mappings
    model, char_to_label, label_to_char = load_model_and_mappings()
    fonts = setup_fonts()
    
    # Create a batch of test images
    test_chars = ['1', '2', '3', 'A', 'B', 'C']
    batch_size = len(test_chars)
    batch_images = np.zeros((batch_size, 32, 32, 1), dtype=np.float32)
    
    print(f"Creating batch of {batch_size} test images...")
    
    for i, char in enumerate(test_chars):
        if char in char_to_label:
            test_image = generate_test_character(char, fonts)
            batch_images[i] = test_image.reshape(32, 32, 1) / 255.0
    
    # Make batch prediction
    print("Making batch prediction...")
    predictions = model.predict(batch_images, verbose=0)
    
    # Process results
    print("Batch prediction results:")
    for i, char in enumerate(test_chars):
        if char in char_to_label:
            predicted_label = np.argmax(predictions[i])
            predicted_char = label_to_char.get(predicted_label, '?')
            confidence = predictions[i][predicted_label]
            
            status = "✓" if predicted_char == char else "✗"
            print(f"  {status} '{char}' -> '{predicted_char}' (confidence: {confidence:.3f})")
    
    print("✅ Batch prediction test completed!")

if __name__ == "__main__":
    print("=== Character Recognition Model Test ===")
    
    try:
        # Test individual predictions
        accuracy = test_model_predictions()
        
        # Test batch prediction
        test_batch_prediction()
        
        print(f"\n✅ All tests completed successfully!")
        print(f"Final accuracy: {accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

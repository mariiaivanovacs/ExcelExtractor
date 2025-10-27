#!/usr/bin/env python3
"""
Test Original Model on Real Data
================================
Test the original working model on real test images from tests/ folder
and compare with the expected sequence: 9, 2, 3, 6, 0, 5, 7, 1
"""

import numpy as np
import cv2
import os
import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_and_preprocess_real_image(image_path, target_size=(32, 32)):
    """
    Load and preprocess a real test image for the model.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the model (32, 32)
    
    Returns:
        Preprocessed image array ready for prediction
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Resize to target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Reshape for model input (batch_size, height, width, channels)
    img = img.reshape(1, target_size[0], target_size[1], 1)
    
    return img

def test_original_model_on_real_data():
    """
    Test the original model on real test images.
    """
    print("=" * 70)
    print("TESTING ORIGINAL MODEL ON REAL DATA")
    print("=" * 70)
    
    # Expected sequence from user
    expected_sequence = [9, 2, 3, 6, 0, 5, 7, 1]
    
    # Try to load the original model
    model_paths = [
        'mnt/numbers/digit_classifier_cnn.keras',
        'mnt/numbers/digit_classifier_cnn.h5',
        'models/improved_results/simple_robust_classifier.keras'
    ]
    
    model = None
    model_name = ""
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"📦 Loading model from {model_path}")
            try:
                model = load_model(model_path)
                model_name = model_path.split('/')[-1].replace('.keras', '').replace('.h5', '')
                break
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                continue
    
    if model is None:
        print("❌ No model found! Please train a model first!")
        return
    
    # Get test images
    test_dir = 'tests'
    image_files = glob.glob(os.path.join(test_dir, '*.png'))
    image_files.sort()  # Sort to get consistent order
    
    if len(image_files) == 0:
        print(f"❌ No PNG images found in {test_dir}/")
        return
    
    print(f"📁 Found {len(image_files)} test images")
    print(f"🎯 Expected sequence: {expected_sequence}")
    print()
    
    # Test each image
    predictions = []
    confidences = []
    
    print("🔍 Testing images:")
    print("-" * 50)
    
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        
        try:
            # Preprocess image
            img = load_and_preprocess_real_image(image_path)
            
            # Get prediction
            pred_proba = model.predict(img, verbose=0)[0]
            predicted_digit = np.argmax(pred_proba)
            confidence = pred_proba[predicted_digit]
            
            predictions.append(predicted_digit)
            confidences.append(confidence)
            
            # Compare with expected if available
            expected = expected_sequence[i] if i < len(expected_sequence) else "?"
            status = "✅" if (i < len(expected_sequence) and predicted_digit == expected) else "❌"
            
            print(f"{i+1:2d}. {filename}")
            print(f"    Expected: {expected} | Predicted: {predicted_digit} | Confidence: {confidence:.4f} {status}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            predictions.append(-1)
            confidences.append(0.0)
    
    print()
    print("=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    
    # Calculate accuracy
    correct = 0
    total = min(len(predictions), len(expected_sequence))
    
    for i in range(total):
        if predictions[i] == expected_sequence[i]:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"Model: {model_name}")
    print(f"Test Images: {len(image_files)}")
    print(f"Expected: {expected_sequence[:total]}")
    print(f"Predicted: {predictions[:total]}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    print()
    
    # Detailed analysis
    print("🔍 Detailed Analysis:")
    print("-" * 30)
    
    for i in range(total):
        expected = expected_sequence[i]
        predicted = predictions[i]
        confidence = confidences[i]
        
        if predicted == expected:
            print(f"✅ Image {i+1}: {expected} → {predicted} (conf: {confidence:.3f}) CORRECT")
        else:
            print(f"❌ Image {i+1}: {expected} → {predicted} (conf: {confidence:.3f}) WRONG")
    
    print()
    
    # Confusion analysis
    if accuracy < 0.8:
        print("⚠️  LOW ACCURACY DETECTED!")
        print("Possible issues:")
        print("- Domain gap between training and real data")
        print("- Different image preprocessing needed")
        print("- Model needs retraining on real data")
        print("- Image quality/resolution mismatch")
    else:
        print("✅ Good accuracy! Model works well on real data.")
    
    print("=" * 70)
    
    return predictions, confidences, accuracy

def visualize_real_test_images():
    """
    Visualize the real test images to understand their characteristics.
    """
    import matplotlib.pyplot as plt
    
    test_dir = 'tests'
    image_files = glob.glob(os.path.join(test_dir, '*.png'))
    image_files.sort()
    
    if len(image_files) == 0:
        print("No test images found!")
        return
    
    # Create visualization
    n_images = min(8, len(image_files))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Real Test Images from tests/ folder', fontsize=16, fontweight='bold')
    
    expected_sequence = [9, 2, 3, 6, 0, 5, 7, 1]
    
    for i in range(n_images):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Load and display image
        img = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        
        filename = os.path.basename(image_files[i])
        expected = expected_sequence[i] if i < len(expected_sequence) else "?"
        ax.set_title(f"{filename}\nExpected: {expected}", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, 8):
        row = i // 4
        col = i % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/real_test_images_visualization.png', dpi=150, bbox_inches='tight')
    print("📸 Real test images visualization saved to: models/real_test_images_visualization.png")
    plt.close()

if __name__ == "__main__":
    # First visualize the real test images
    print("📸 Visualizing real test images...")
    visualize_real_test_images()
    print()
    
    # Then test the model
    predictions, confidences, accuracy = test_original_model_on_real_data()
    
    print(f"\n🎯 Final Result: {accuracy:.1%} accuracy on real test data")
    
    if accuracy < 0.5:
        print("\n💡 Recommendations:")
        print("1. Check if the original model exists and is trained")
        print("2. Verify image preprocessing matches training data")
        print("3. Consider retraining with real data samples")
        print("4. Check if images need different preprocessing")

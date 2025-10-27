#!/usr/bin/env python3
"""
Test Robust Model with Distortion
=================================
Test the robust model with your original downsample function to see
how it performs on distorted images similar to your use case.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import warnings
import os
import cv2
warnings.filterwarnings('ignore')

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

def load_system_fonts():
    """Load available system fonts for digit generation."""
    fonts = []
    
    try:
        # Try to load some common fonts
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 20)
                    fonts.append(font)
                    break
                except:
                    continue
        
        # Fallback to default font
        if not fonts:
            fonts.append(ImageFont.load_default())
            
    except Exception as e:
        fonts.append(ImageFont.load_default())
    
    return fonts


def generate_clean_digit(digit, font, img_size=32):
    """Generate a clean digit image."""
    # Create white background
    img = Image.new('L', (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    
    # Draw digit in black
    text = str(digit)
    
    # Get text size and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (img_size - text_width) // 2
    y = (img_size - text_height) // 2
    
    draw.text((x, y), text, fill=0, font=font)
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    return img_array


def downsample_then_upsample(img, scale_range=(0.3, 0.7)):
    """
    Your original downsample function - simulates poor quality images.
    """
    h, w = img.shape
    scale = np.random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Downsample
    downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Add Gaussian blur
    kernel_size = random.choice([3, 5, 7])
    blurred = cv2.GaussianBlur(downsampled, (kernel_size, kernel_size), 0)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, blurred.shape)
    noisy = np.clip(blurred + noise, 0, 1)
    
    # Upsample back
    upsampled = cv2.resize(noisy, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_uint8 = (upsampled * 255).astype(np.uint8)
    enhanced = clahe.apply(img_uint8)
    result = enhanced.astype(np.float32) / 255.0
    
    # Additional distortions
    if random.random() < 0.3:
        # Random erosion/dilation
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            result_uint8 = (result * 255).astype(np.uint8)
            result_uint8 = cv2.erode(result_uint8, kernel, iterations=1)
            result = result_uint8.astype(np.float32) / 255.0
        else:
            result_uint8 = (result * 255).astype(np.uint8)
            result_uint8 = cv2.dilate(result_uint8, kernel, iterations=1)
            result = result_uint8.astype(np.float32) / 255.0
    
    # Random dimming
    if random.random() < 0.2:
        dim_factor = np.random.uniform(0.3, 0.7)
        result = result * dim_factor
    
    return result


def create_test_dataset(fonts, samples_per_digit=100, img_size=32, apply_distortion=True):
    """Create test dataset with optional distortion."""
    print(f"\nGenerating test dataset...")
    print(f"  Samples per digit: {samples_per_digit}")
    print(f"  Apply distortion: {apply_distortion}")
    
    X = []
    y = []
    
    for digit in range(10):
        print(f"  Generating digit {digit}...")
        for _ in range(samples_per_digit):
            font = random.choice(fonts)
            
            # Generate clean digit
            img = generate_clean_digit(digit, font, img_size)
            
            # Apply distortion if requested
            if apply_distortion:
                img = downsample_then_upsample(img)
            
            X.append(img.reshape(img_size, img_size, 1))
            y.append(digit)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Test dataset created: {X.shape}")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")
    
    return X, y


def evaluate_model_on_distortion(model, X_test, y_test, dataset_name):
    """Evaluate model and show detailed results."""
    print(f"\n" + "=" * 60)
    print(f"EVALUATION ON {dataset_name.upper()}")
    print("=" * 60)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✓ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for digit in range(10):
        mask = y_test == digit
        if np.sum(mask) > 0:
            digit_accuracy = np.mean(y_pred[mask] == digit)
            print(f"  Digit {digit}: {digit_accuracy:.4f} ({digit_accuracy*100:.1f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print("    Predicted:")
    print("      ", end="")
    for i in range(10):
        print(f"{i:4d}", end="")
    print()
    
    for i in range(10):
        print(f" {i}: ", end="")
        for j in range(10):
            print(f"{cm[i,j]:4d}", end="")
        print()
    
    # Analyze specific confusion patterns
    print(f"\n🎯 Confusion Analysis:")
    problematic_pairs = [(3, 9), (5, 6), (0, 8)]
    
    for digit1, digit2 in problematic_pairs:
        confusion_12 = cm[digit1, digit2]  # digit1 predicted as digit2
        confusion_21 = cm[digit2, digit1]  # digit2 predicted as digit1
        
        if confusion_12 > 0 or confusion_21 > 0:
            print(f"  {digit1}→{digit2}: {confusion_12} errors")
            print(f"  {digit2}→{digit1}: {confusion_21} errors")
        else:
            print(f"  ✅ No confusion between {digit1} and {digit2}")
    
    return accuracy, cm


def visualize_sample_predictions(model, X_test, y_test, num_samples=10):
    """Visualize some sample predictions."""
    print(f"\n🖼️  Sample Predictions:")
    
    # Get predictions
    y_pred_proba = model.predict(X_test[:num_samples], verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        true_label = y_test[i]
        pred_label = y_pred[i]
        confidence = y_pred_proba[i, pred_label]
        
        axes[i].imshow(X_test[i].squeeze(), cmap='gray')
        
        if true_label == pred_label:
            color = 'green'
            title = f'✓ True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}'
        else:
            color = 'red'
            title = f'✗ True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}'
        
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/improved_results/sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample predictions saved to: models/improved_results/sample_predictions.png")


def main():
    """Main testing pipeline."""
    print("=" * 70)
    print("TESTING ROBUST MODEL WITH DISTORTION")
    print("=" * 70)
    
    # Load the robust model
    model_path = 'models/improved_results/simple_robust_classifier.keras'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run train_simple_robust.py first!")
        return
    
    print(f"📦 Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load fonts
    fonts = load_system_fonts()
    
    # Test 1: Clean images (should be 100% accurate)
    print("\n" + "🧪" * 3 + " TEST 1: CLEAN IMAGES " + "🧪" * 3)
    X_clean, y_clean = create_test_dataset(fonts, samples_per_digit=50, apply_distortion=False)
    acc_clean, cm_clean = evaluate_model_on_distortion(model, X_clean, y_clean, "Clean Images")
    
    # Test 2: Distorted images (your original downsample function)
    print("\n" + "🧪" * 3 + " TEST 2: DISTORTED IMAGES " + "🧪" * 3)
    X_distorted, y_distorted = create_test_dataset(fonts, samples_per_digit=50, apply_distortion=True)
    acc_distorted, cm_distorted = evaluate_model_on_distortion(model, X_distorted, y_distorted, "Distorted Images")
    
    # Visualize some sample predictions
    visualize_sample_predictions(model, X_distorted, y_distorted, num_samples=10)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Clean Images Accuracy:     {acc_clean:.4f} ({acc_clean*100:.2f}%)")
    print(f"Distorted Images Accuracy: {acc_distorted:.4f} ({acc_distorted*100:.2f}%)")
    print(f"Performance Drop:          {(acc_clean - acc_distorted):.4f} ({(acc_clean - acc_distorted)*100:.2f}%)")
    
    if acc_distorted > 0.8:
        print("🎉 Excellent performance on distorted images!")
    elif acc_distorted > 0.6:
        print("👍 Good performance on distorted images!")
    elif acc_distorted > 0.4:
        print("⚠️  Moderate performance - may need improvement")
    else:
        print("❌ Poor performance - needs significant improvement")


if __name__ == "__main__":
    main()

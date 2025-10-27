#!/usr/bin/env python3
"""
Test Models with Proper Preprocessing
=====================================
Test models on real data with proper preprocessing (image inversion)
to match the training data polarity.
"""

import numpy as np
import cv2
import os
import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_real_image_properly(image_path, target_size=(32, 32)):
    """
    Properly preprocess real images to match training data.
    
    Key insight: Real images have dark text on light background,
    but models were trained on light text on dark background.
    Solution: INVERT the images!
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # CRITICAL: Invert the image (dark text → light text)
    img = 255 - img
    
    # Optional: Apply thresholding for cleaner binary image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Resize to target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Reshape for model input
    img = img.reshape(1, target_size[0], target_size[1], 1)
    
    return img

def test_all_models_with_proper_preprocessing():
    """
    Test all available models with proper preprocessing.
    """
    print("=" * 70)
    print("TESTING MODELS WITH PROPER PREPROCESSING (IMAGE INVERSION)")
    print("=" * 70)
    
    # Expected sequence from user
    expected_sequence = [9, 2, 3, 6, 0, 5, 7, 1]
    
    # Try different models
    model_paths = [
        ('Original CNN', 'mnt/numbers/digit_classifier_cnn.keras'),
        ('Simple Robust', 'models/improved_results/simple_robust_classifier.keras'),
        ('Distortion Robust', 'models/improved_results/distortion_robust_classifier.keras'),
        ('Targeted Robust', 'models/improved_results/targeted_robust_final.keras'),
        ('Targeted Best', 'models/improved_results/targeted_robust_best.keras')
    ]
    
    # Get test images
    test_dir = 'tests'
    image_files = glob.glob(os.path.join(test_dir, '*.png'))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"❌ No PNG images found in {test_dir}/")
        return
    
    print(f"📁 Found {len(image_files)} test images")
    print(f"🎯 Expected sequence: {expected_sequence}")
    print()
    
    results = {}
    
    # Test each model
    for model_name, model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"⏭️  Skipping {model_name}: Model not found")
            continue
        
        print(f"🧪 Testing {model_name}...")
        print(f"   Model: {model_path}")
        
        try:
            # Load model
            model = load_model(model_path)
            
            # Test on all images
            predictions = []
            confidences = []
            
            for i, image_path in enumerate(image_files):
                try:
                    # Preprocess with proper inversion
                    img = preprocess_real_image_properly(image_path)
                    
                    # Get prediction
                    pred_proba = model.predict(img, verbose=0)[0]
                    predicted_digit = np.argmax(pred_proba)
                    confidence = pred_proba[predicted_digit]
                    
                    predictions.append(predicted_digit)
                    confidences.append(confidence)
                    
                except Exception as e:
                    print(f"   ❌ Error processing image {i+1}: {e}")
                    predictions.append(-1)
                    confidences.append(0.0)
            
            # Calculate accuracy
            correct = 0
            total = min(len(predictions), len(expected_sequence))
            
            for i in range(total):
                if predictions[i] == expected_sequence[i]:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            avg_confidence = np.mean([c for c in confidences if c > 0])
            
            results[model_name] = {
                'predictions': predictions[:total],
                'confidences': confidences[:total],
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'correct': correct,
                'total': total
            }
            
            print(f"   ✅ Accuracy: {correct}/{total} = {accuracy:.1%}")
            print(f"   📊 Avg Confidence: {avg_confidence:.3f}")
            print(f"   🎯 Predictions: {predictions[:total]}")
            print()
            
        except Exception as e:
            print(f"   ❌ Failed to load/test {model_name}: {e}")
            print()
    
    # Summary comparison
    print("=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    
    if results:
        print(f"{'Model':<20} {'Accuracy':<10} {'Avg Conf':<10} {'Predictions'}")
        print("-" * 70)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['accuracy']:<10.1%} "
                  f"{result['avg_confidence']:<10.3f} {result['predictions']}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print()
        print(f"🏆 BEST MODEL: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy")
        
        # Detailed analysis of best model
        print()
        print("🔍 DETAILED ANALYSIS OF BEST MODEL:")
        print("-" * 40)
        
        best_result = best_model[1]
        for i in range(best_result['total']):
            expected = expected_sequence[i]
            predicted = best_result['predictions'][i]
            confidence = best_result['confidences'][i]
            
            status = "✅" if predicted == expected else "❌"
            print(f"Image {i+1}: {expected} → {predicted} (conf: {confidence:.3f}) {status}")
        
        print()
        print("🎯 KEY INSIGHTS:")
        print("-" * 20)
        if best_result['accuracy'] > 0.7:
            print("✅ EXCELLENT! Image inversion solved the problem!")
            print("✅ The issue was indeed polarity mismatch")
            print("✅ Models work well when images are properly preprocessed")
        elif best_result['accuracy'] > 0.4:
            print("⚠️  GOOD IMPROVEMENT! Image inversion helped significantly")
            print("⚠️  Some confusion remains - may need model fine-tuning")
        else:
            print("❌ Still low accuracy - may need different approach")
            print("❌ Consider retraining with real data samples")
    
    else:
        print("❌ No models could be tested!")
    
    print("=" * 70)
    
    return results

def create_before_after_visualization():
    """
    Create visualization showing before/after preprocessing.
    """
    import matplotlib.pyplot as plt
    
    test_dir = 'tests'
    image_files = glob.glob(os.path.join(test_dir, '*.png'))
    image_files.sort()
    
    if len(image_files) == 0:
        return
    
    # Show first 4 images
    n_images = min(4, len(image_files))
    fig, axes = plt.subplots(2, n_images, figsize=(16, 8))
    
    expected_sequence = [9, 2, 3, 6, 0, 5, 7, 1]
    
    for i in range(n_images):
        # Original image
        img_orig = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img_orig, cmap='gray')
        axes[0, i].set_title(f'Original\nExpected: {expected_sequence[i]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Preprocessed image (inverted + thresholded)
        img_proc = 255 - img_orig
        _, img_proc = cv2.threshold(img_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        axes[1, i].imshow(img_proc, cmap='gray')
        axes[1, i].set_title('Inverted + Thresholded\n(Model Input)', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('Before/After Preprocessing for Model Input', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/before_after_preprocessing.png', dpi=150, bbox_inches='tight')
    print("📸 Before/after visualization saved to: models/before_after_preprocessing.png")
    plt.close()

if __name__ == "__main__":
    # Create before/after visualization
    print("📸 Creating before/after preprocessing visualization...")
    create_before_after_visualization()
    print()
    
    # Test all models with proper preprocessing
    results = test_all_models_with_proper_preprocessing()
    
    print("\n💡 SOLUTION SUMMARY:")
    print("=" * 50)
    print("🔑 ROOT CAUSE: Image polarity mismatch")
    print("   - Models trained on: Light text on dark background")
    print("   - Real images have: Dark text on light background")
    print()
    print("🔧 SOLUTION: Invert images before prediction")
    print("   - Apply: img = 255 - img")
    print("   - Optional: Add OTSU thresholding")
    print()
    print("✅ This should dramatically improve accuracy!")
    print("=" * 50)

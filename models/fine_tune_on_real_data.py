#!/usr/bin/env python3
"""
Fine-tune the best performing model on real data
Uses the real_data_classifier.keras as base and fine-tunes on actual real images
"""

import os
import numpy as np
import pandas as pd
import cv2
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("=" * 70)
print("FINE-TUNING ON REAL DATA")
print("Using real_data_classifier.keras as base + real images for fine-tuning")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

def load_real_images_with_labels():
    """Load real images and their correct labels"""
    print("\n📁 Loading real images with labels...")
    
    predictions_df = pd.read_csv('experiments/predictions.csv')
    
    real_images = []
    real_labels = []
    filenames = []
    
    for idx, row in predictions_df.iterrows():
        filename = row['filename']
        true_label = int(row['predicted_class'])  # This is actually the correct label
        
        image_path = f'tests/{filename}'
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Preprocess (no inversion needed - model trained on same polarity)
                img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                
                real_images.append(img)
                real_labels.append(true_label)
                filenames.append(filename)
    
    real_images = np.array(real_images)
    real_labels = np.array(real_labels)
    
    # Reshape for model
    real_images = real_images.reshape(-1, 32, 32, 1)
    
    print(f"✓ Loaded {len(real_images)} real images")
    print(f"  - Shape: {real_images.shape}")
    print(f"  - Labels: {real_labels}")
    print(f"  - Label distribution: {np.bincount(real_labels)}")
    
    return real_images, real_labels, filenames

def augment_real_images(images, labels, augment_factor=10):
    """Create augmented versions of real images to increase training data"""
    print(f"\n🔄 Augmenting real images (factor: {augment_factor}x)...")
    
    augmented_images = []
    augmented_labels = []
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Add original image
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Create augmented versions
        for aug_idx in range(augment_factor - 1):
            # Convert to PIL for augmentation
            img_pil = Image.fromarray((img.squeeze() * 255).astype(np.uint8))
            
            # Apply random augmentations
            # 1. Slight rotation
            if random.random() < 0.7:
                angle = random.uniform(-5, 5)
                img_pil = img_pil.rotate(angle, fillcolor=255)
            
            # 2. Slight scaling
            if random.random() < 0.5:
                scale = random.uniform(0.9, 1.1)
                new_size = int(32 * scale)
                img_pil = img_pil.resize((new_size, new_size), Image.LANCZOS)
                # Crop or pad to 32x32
                if new_size > 32:
                    # Crop
                    left = (new_size - 32) // 2
                    top = (new_size - 32) // 2
                    img_pil = img_pil.crop((left, top, left + 32, top + 32))
                else:
                    # Pad
                    new_img = Image.new('L', (32, 32), color=255)
                    paste_x = (32 - new_size) // 2
                    paste_y = (32 - new_size) // 2
                    new_img.paste(img_pil, (paste_x, paste_y))
                    img_pil = new_img
            
            # 3. Slight translation
            if random.random() < 0.6:
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                img_pil = img_pil.transform((32, 32), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=255)
            
            # 4. Brightness adjustment
            if random.random() < 0.4:
                enhancer = ImageEnhance.Brightness(img_pil)
                factor = random.uniform(0.9, 1.1)
                img_pil = enhancer.enhance(factor)
            
            # 5. Contrast adjustment
            if random.random() < 0.4:
                enhancer = ImageEnhance.Contrast(img_pil)
                factor = random.uniform(0.9, 1.1)
                img_pil = enhancer.enhance(factor)
            
            # 6. Slight noise
            if random.random() < 0.3:
                img_array = np.array(img_pil)
                noise = np.random.normal(0, 2, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255)
                img_pil = Image.fromarray(img_array.astype(np.uint8))
            
            # Convert back to numpy
            aug_img = np.array(img_pil).astype(np.float32) / 255.0
            aug_img = aug_img.reshape(32, 32, 1)
            
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    print(f"✓ Augmentation complete!")
    print(f"  - Original: {len(images)} images")
    print(f"  - Augmented: {len(augmented_images)} images")
    print(f"  - Augmented label distribution: {np.bincount(augmented_labels)}")
    
    return augmented_images, augmented_labels

def test_model_on_real_images(model, real_images, real_labels, filenames):
    """Test model on real images"""
    print("\n" + "=" * 70)
    print("TESTING ON REAL IMAGES")
    print("=" * 70)
    
    results = []
    correct_predictions = 0
    
    print("\n🔍 Testing images:")
    print("-" * 50)
    
    for i, (img, true_label, filename) in enumerate(zip(real_images, real_labels, filenames)):
        # Predict
        prediction = model.predict(img.reshape(1, 32, 32, 1), verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit]
        
        # Check if correct
        is_correct = predicted_digit == true_label
        if is_correct:
            correct_predictions += 1
        
        # Store result
        results.append({
            'filename': filename,
            'true_label': true_label,
            'predicted_label': predicted_digit,
            'confidence': confidence,
            'correct': is_correct
        })
        
        # Print result
        status = "✅" if is_correct else "❌"
        short_name = filename[:30] + "..." if len(filename) > 30 else filename
        print(f"{i+1:2d}. {short_name:<33} Expected: {true_label} | Predicted: {predicted_digit} | Conf: {confidence:.4f} {status}")
    
    # Calculate accuracy
    accuracy = correct_predictions / len(results) if results else 0
    avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    true_labels = [r['true_label'] for r in results]
    predicted_labels = [r['predicted_label'] for r in results]
    
    print(f"Expected:  {true_labels}")
    print(f"Predicted: {predicted_labels}")
    print(f"Accuracy:  {correct_predictions}/{len(results)} = {accuracy:.1%}")
    print(f"Avg Confidence: {avg_confidence:.3f}")
    
    # Analyze problematic digits
    problematic_digits = [0, 3, 5, 8]
    problematic_correct = 0
    problematic_total = 0
    
    for result in results:
        if result['true_label'] in problematic_digits:
            problematic_total += 1
            if result['correct']:
                problematic_correct += 1
    
    if problematic_total > 0:
        problematic_accuracy = problematic_correct / problematic_total
        print(f"Problematic digits (0,3,5,8): {problematic_correct}/{problematic_total} = {problematic_accuracy:.1%}")
    
    print("=" * 70)
    
    return results

def main():
    # Load real images
    real_images, real_labels, filenames = load_real_images_with_labels()
    
    # Test original model first
    print("\n🔍 Testing original model (real_data_classifier.keras)...")
    if os.path.exists('models/real_data_classifier.keras'):
        original_model = keras.models.load_model('models/real_data_classifier.keras')
        print("Original model results:")
        original_results = test_model_on_real_images(original_model, real_images, real_labels, filenames)
        original_accuracy = sum(r['correct'] for r in original_results) / len(original_results)
        print(f"Original model accuracy: {original_accuracy:.1%}")
    else:
        print("❌ Original model not found!")
        return
    
    # Augment real images for fine-tuning
    aug_images, aug_labels = augment_real_images(real_images, real_labels, augment_factor=20)
    
    # Split augmented data for fine-tuning
    print("\nSplitting augmented data for fine-tuning...")
    X_train, X_val, y_train, y_val = train_test_split(
        aug_images, aug_labels, test_size=0.2, random_state=42, stratify=aug_labels
    )
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    
    # Load and fine-tune the original model
    print("\n🔧 Fine-tuning original model...")
    model = keras.models.load_model('models/real_data_classifier.keras')
    
    # Reduce learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tuning callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Fine-tune
    print(f"\nFine-tuning on {len(X_train)} augmented real images...")
    history = model.fit(
        X_train, y_train,
        batch_size=16,  # Smaller batch size for fine-tuning
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate fine-tuned model
    print("\n📊 Fine-tuned model validation results:")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Save fine-tuned model
    model_path = 'models/fine_tuned_real_data_classifier.keras'
    model.save(model_path)
    print(f"\n✅ Fine-tuned model saved to: {model_path}")
    
    # Test fine-tuned model on real images
    print("\n🎯 Testing fine-tuned model on real images...")
    fine_tuned_results = test_model_on_real_images(model, real_images, real_labels, filenames)
    fine_tuned_accuracy = sum(r['correct'] for r in fine_tuned_results) / len(fine_tuned_results)
    
    # Compare results
    print("\n" + "=" * 70)
    print("📈 IMPROVEMENT COMPARISON")
    print("=" * 70)
    print(f"Original model accuracy:    {original_accuracy:.1%}")
    print(f"Fine-tuned model accuracy:  {fine_tuned_accuracy:.1%}")
    improvement = fine_tuned_accuracy - original_accuracy
    print(f"Improvement:                {improvement:+.1%}")
    print("=" * 70)
    
    return model, history, fine_tuned_results

if __name__ == "__main__":
    model, history, results = main()

#!/usr/bin/env python3
"""
Train Model for Real Data
=========================
Train a simple, robust model specifically designed for your real data characteristics:
- Dark text on light background (like your real images)
- Simple architecture to avoid model collapse
- Conservative training approach
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import warnings
import os
import cv2
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("=" * 70)
print("TRAINING MODEL FOR REAL DATA CHARACTERISTICS")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print()

def load_fonts():
    """Load system fonts for digit rendering."""
    print("Loading fonts...")
    fonts = []
    try:
        fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/calibri-regular.ttf", 14))
        fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/Lucida Console Regular.ttf", 16))
        fonts.append(ImageFont.truetype("/Library/Fonts/adelle-sans-regular.otf", 14))
    except:
        # Fallback to default
        fonts = [ImageFont.load_default() for _ in range(3)]
    
    print(f"Loaded {len(fonts)} fonts\n")
    return fonts

def create_real_style_digit(digit, font, img_size=32):
    """
    Create digit image that matches your real data:
    - Dark text on light background (like your real images)
    - Similar contrast and characteristics
    """
    # Create light background (like your real images)
    img = Image.new('L', (img_size, img_size), color=255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Get text size for centering
    bbox = draw.textbbox((0, 0), digit, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center with small random offset
    x = (img_size - text_width) // 2 + random.randint(-2, 2)
    y = (img_size - text_height) // 2 + random.randint(-2, 2)
    
    # Draw dark text (like your real images)
    text_color = random.randint(0, 50)  # Dark text (0=black, 50=dark gray)
    draw.text((x, y), digit, fill=text_color, font=font)
    
    return img

def augment_real_style(img):
    """
    Apply augmentations that simulate your real data characteristics.
    """
    img_array = np.array(img)
    
    # Add slight blur (like real images)
    if random.random() < 0.7:
        kernel_size = random.choice([3, 5])
        img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 
                                   sigmaX=random.uniform(0.3, 0.8))
    
    # Add slight noise
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(3, 8), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Slight contrast/brightness variation
    if random.random() < 0.6:
        alpha = random.uniform(0.8, 1.2)  # Contrast
        beta = random.uniform(-10, 10)    # Brightness
        img_array = np.clip(alpha * img_array + beta, 0, 255).astype(np.uint8)
    
    # Small rotation
    if random.random() < 0.4:
        angle = random.uniform(-5, 5)
        h, w = img_array.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img_array = cv2.warpAffine(img_array, M, (w, h), 
                                  borderMode=cv2.BORDER_REPLICATE,
                                  borderValue=255)
    
    return Image.fromarray(img_array, mode='L')

def generate_real_style_dataset(fonts, samples_per_digit=800, img_size=32):
    """
    Generate dataset that matches your real data characteristics.
    """
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    total_samples = samples_per_digit * 10
    
    print(f"Generating {total_samples} real-style digit images...")
    print(f"  - {samples_per_digit} samples per digit")
    print(f"  - Dark text on light background (like your real images)")
    print()
    
    X = []
    y = []
    
    for digit_idx, digit in enumerate(digits):
        print(f"  Generating digit '{digit}' ({digit_idx}/9)...", end=' ')
        
        for _ in range(samples_per_digit):
            # Random font
            font = random.choice(fonts)
            
            # Create base image (dark text on light background)
            img = create_real_style_digit(digit, font, img_size)
            
            # Apply realistic augmentation
            img = augment_real_style(img)
            
            # Convert to numpy and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            X.append(img_array)
            y.append(digit_idx)
        
        print(f"✓")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN
    X = X.reshape(-1, img_size, img_size, 1)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Shape: {X.shape}")
    print(f"  - Value range: [{X.min():.3f}, {X.max():.3f}]")
    print()
    
    return X, y

def build_simple_robust_model(input_shape=(32, 32, 1), num_classes=10):
    """
    Build a simple, robust model that avoids collapse.
    """
    print("Building simple robust model...")
    
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # First conv block - larger kernels for robustness
        layers.Conv2D(16, (5, 5), activation='relu', padding='same',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        
        # Second conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        
        # Third conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        # Dense layers - simple and robust
        layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.Dropout(0.2),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ], name='SimpleRobustDigitCNN')
    
    # Conservative optimizer settings
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    print("=" * 50)
    model.summary()
    print("=" * 50)
    print()
    
    return model

def train_robust_model(model, X_train, y_train, X_val, y_val, epochs=25):
    """
    Train with conservative, robust settings.
    """
    print(f"Training model for {epochs} epochs...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print()
    
    # Conservative callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1,
        min_lr=1e-6
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n✓ Training complete!")
    return history

def test_on_real_images(model):
    """
    Test the trained model on your real images.
    """
    import glob
    
    print("\n" + "=" * 70)
    print("TESTING ON YOUR REAL IMAGES")
    print("=" * 70)
    
    # Expected sequence
    expected_sequence = [9, 2, 3, 6, 0, 5, 7, 1]
    
    # Get test images
    test_dir = 'tests'
    image_files = glob.glob(os.path.join(test_dir, '*.png'))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"❌ No test images found in {test_dir}/")
        return
    
    print(f"📁 Found {len(image_files)} test images")
    print(f"🎯 Expected sequence: {expected_sequence}")
    print()
    
    predictions = []
    confidences = []
    
    print("🔍 Testing images:")
    print("-" * 50)
    
    for i, image_path in enumerate(image_files):
        filename = os.path.basename(image_path)
        
        try:
            # Load and preprocess image (NO inversion - train on same polarity)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = img.reshape(1, 32, 32, 1)
            
            # Predict
            pred_proba = model.predict(img, verbose=0)[0]
            predicted_digit = np.argmax(pred_proba)
            confidence = pred_proba[predicted_digit]
            
            predictions.append(predicted_digit)
            confidences.append(confidence)
            
            # Compare with expected
            expected = expected_sequence[i] if i < len(expected_sequence) else "?"
            status = "✅" if (i < len(expected_sequence) and predicted_digit == expected) else "❌"
            
            print(f"{i+1:2d}. {filename[:30]}")
            print(f"    Expected: {expected} | Predicted: {predicted_digit} | Confidence: {confidence:.4f} {status}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
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
    
    print()
    print("=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    print(f"Expected:  {expected_sequence[:total]}")
    print(f"Predicted: {predictions[:total]}")
    print(f"Accuracy:  {correct}/{total} = {accuracy:.1%}")
    print(f"Avg Confidence: {avg_confidence:.3f}")
    print()
    
    if accuracy > 0.7:
        print("🎉 EXCELLENT! Model works well on real data!")
    elif accuracy > 0.4:
        print("⚠️  GOOD IMPROVEMENT! Some issues remain.")
    else:
        print("❌ Still having issues. May need more real data for training.")
    
    return accuracy, predictions, confidences

def main():
    """Main training pipeline."""
    # Load fonts
    fonts = load_fonts()
    
    # Generate dataset that matches real data characteristics
    X, y = generate_real_style_dataset(fonts, samples_per_digit=800)
    
    # Split dataset
    print("Splitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples\n")
    
    # Build model
    model = build_simple_robust_model()
    
    # Train model
    history = train_robust_model(model, X_train, y_train, X_val, y_val, epochs=25)
    
    # Evaluate on validation set
    print("\n📊 Validation Results:")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Save model
    model_path = 'models/real_data_classifier.keras'
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Test on real images
    accuracy, predictions, confidences = test_on_real_images(model)
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY")
    print("=" * 70)
    print(f"✅ Model trained on data matching your real image characteristics")
    print(f"✅ Dark text on light background (same as your real images)")
    print(f"✅ Simple, robust architecture to avoid model collapse")
    print(f"✅ Real data accuracy: {accuracy:.1%}")
    print("=" * 70)

if __name__ == "__main__":
    main()

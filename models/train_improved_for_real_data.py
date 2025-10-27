#!/usr/bin/env python3
"""
Enhanced CNN Training for Real Data - Focused on Problematic Digits
Addresses specific confusions: 3, 5, 8, 0
"""

import os
import numpy as np
import pandas as pd
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("=" * 70)
print("ENHANCED CNN TRAINING FOR REAL DATA")
print("Focusing on problematic digits: 3, 5, 8, 0")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

class EnhancedDigitGenerator:
    def __init__(self, img_size=32):
        self.img_size = img_size
        self.fonts = self.load_fonts()
        
    def load_fonts(self):
        """Load available fonts"""
        print("\nLoading fonts...")
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc", 
            "/System/Library/Fonts/Times.ttc"
        ]
        
        fonts = []
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    # Try different sizes for variety
                    for size in [20, 22, 24, 26]:
                        fonts.append(ImageFont.truetype(font_path, size))
                except:
                    pass
        
        if not fonts:
            fonts = [ImageFont.load_default()]
        
        print(f"Loaded {len(fonts)} font variations")
        return fonts
    
    def create_real_style_digit(self, digit, font, add_noise=True):
        """Create digit image matching real data characteristics"""
        # Create light background (like real images)
        img = Image.new('L', (self.img_size, self.img_size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Dark text color with variation
        text_color = random.randint(0, 60)  # Dark text
        
        # Calculate text position for centering
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (self.img_size - text_width) // 2 + random.randint(-2, 2)
        y = (self.img_size - text_height) // 2 + random.randint(-2, 2)
        
        # Draw the digit
        draw.text((x, y), str(digit), fill=text_color, font=font)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Add realistic noise and variations
        if add_noise:
            # Add slight gaussian noise
            noise = np.random.normal(0, 3, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            
            # Add slight blur occasionally
            if random.random() < 0.3:
                img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
        
        return img_array.astype(np.uint8)
    
    def generate_enhanced_dataset(self, samples_per_digit=1000, problematic_boost=2.0):
        """Generate dataset with extra focus on problematic digits"""
        print(f"\nGenerating enhanced dataset...")
        print(f"- Base samples per digit: {samples_per_digit}")
        print(f"- Problematic digits (3,5,8,0) boost: {problematic_boost}x")
        
        problematic_digits = [0, 3, 5, 8]
        
        all_images = []
        all_labels = []
        
        for digit in range(10):
            # Calculate samples for this digit
            if digit in problematic_digits:
                num_samples = int(samples_per_digit * problematic_boost)
                print(f"  Generating digit '{digit}' ({num_samples} samples - BOOSTED) ✓")
            else:
                num_samples = samples_per_digit
                print(f"  Generating digit '{digit}' ({num_samples} samples) ✓")
            
            for i in range(num_samples):
                font = random.choice(self.fonts)
                img = self.create_real_style_digit(digit, font)
                
                all_images.append(img)
                all_labels.append(digit)
        
        # Convert to numpy arrays
        X = np.array(all_images)
        y = np.array(all_labels)
        
        # Normalize and reshape
        X = X.astype(np.float32) / 255.0
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  - Total samples: {len(X)}")
        print(f"  - Shape: {X.shape}")
        print(f"  - Value range: [{X.min():.3f}, {X.max():.3f}]")
        
        return X, y

def build_enhanced_cnn():
    """Build enhanced CNN architecture focused on digit discrimination"""
    model = keras.Sequential([
        # First conv block - larger kernels for better feature extraction
        keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', 
                           input_shape=(32, 32, 1), name='conv1'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Second conv block - focus on fine details
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Third conv block - high-level features
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Fourth conv block - very fine features for digit discrimination
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers with strong regularization
        keras.layers.Dense(512, activation='relu', name='dense1'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(256, activation='relu', name='dense2'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(128, activation='relu', name='dense3'),
        keras.layers.Dropout(0.3),
        
        # Output layer
        keras.layers.Dense(10, activation='softmax', name='output')
    ], name='EnhancedDigitCNN')
    
    return model

def test_on_real_images():
    """Test the trained model on real images"""
    print("\n" + "=" * 70)
    print("TESTING ON REAL IMAGES")
    print("=" * 70)
    
    # Load the ground truth
    predictions_df = pd.read_csv('experiments/predictions.csv')
    
    # Load the trained model
    model = keras.models.load_model('models/enhanced_real_data_classifier.keras')
    
    print(f"📁 Found {len(predictions_df)} test images")
    print("🎯 Testing with ground truth labels...")
    
    results = []
    correct_predictions = 0
    
    print("\n🔍 Testing images:")
    print("-" * 50)
    
    for idx, row in predictions_df.iterrows():
        filename = row['filename']
        true_label = int(row['predicted_class'])  # This is actually the correct label
        
        # Load and preprocess image
        image_path = f'tests/{filename}'
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
            
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Could not load: {image_path}")
            continue
        
        # Preprocess (no inversion needed - model trained on same polarity)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = img.reshape(1, 32, 32, 1)
        
        # Predict
        prediction = model.predict(img, verbose=0)
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
        print(f"{idx+1:2d}. {short_name:<33} Expected: {true_label} | Predicted: {predicted_digit} | Conf: {confidence:.4f} {status}")
    
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
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Generate enhanced dataset
    generator = EnhancedDigitGenerator()
    X, y = generator.generate_enhanced_dataset(samples_per_digit=800, problematic_boost=2.5)
    
    # Split dataset
    print("\nSplitting dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    
    # Build model
    print("\nBuilding enhanced CNN...")
    model = build_enhanced_cnn()
    
    # Compile with class weights for problematic digits
    class_weights = {i: 1.0 for i in range(10)}
    for digit in [0, 3, 5, 8]:  # Boost problematic digits
        class_weights[digit] = 2.0
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    print("=" * 50)
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nTraining model with class weights: {class_weights}")
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Validation Results:")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    # Save model
    model_path = 'models/enhanced_real_data_classifier.keras'
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Test on real images
    test_results = test_on_real_images()
    
    return model, history, test_results

if __name__ == "__main__":
    model, history, results = main()

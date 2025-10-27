#!/usr/bin/env python3
"""
IMPROVED CNN Digit Classifier: High-Accuracy Digit Recognition (0-9)
====================================================================
This script creates an improved CNN model for digit recognition with better
preprocessing, architecture, and training strategies.

Key Improvements:
- Better data preprocessing (less aggressive distortion)
- Improved CNN architecture with more layers
- Progressive training strategy
- Better data augmentation
- Ensemble techniques
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
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("=" * 70)
print("IMPROVED DIGIT CNN CLASSIFIER (0-9) - HIGH ACCURACY VERSION")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print()

def load_system_fonts():
    """Load available system fonts for digit rendering."""
    print("Loading system fonts...")
    font_size = 12
    fonts = []
    # Try to use a simple font
    font = ImageFont.truetype("/Users/mariaivanova/Library/Fonts/calibri-regular.ttf")
    fonts.append(font)
    fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/Lucida Console Regular.ttf"))
    # # fonts.append(font)
    font=ImageFont.truetype("/Library/Fonts/adelle-sans-regular.otf")
    fonts.append(font)
    
    font = ImageFont.truetype("/Users/mariaivanova/Downloads/rainyhearts.ttf")
    fonts.append(font)
    font_sizes = [12, 14, 16, 18, 20, 22]
    
    print(f"Loaded {len(fonts)} font variations\n")
    return fonts


def mask_pixels(img):
    import numpy as np
    import cv2

    # Convert to float 0–1
    img = img.astype(np.float32) / 255.0

    # If image is grayscale (2D), expand to 3D so we can use axis=2
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)

    # Compute brightness and color variation
    brightness = np.mean(img, axis=2)
    color_diff = np.max(img, axis=2) - np.min(img, axis=2)

    # Threshold
    bright_thresh = 0.83
    color_thresh = 0.1

    # Mask for light gray pixels
    mask = (brightness > bright_thresh) & (color_diff < color_thresh)

    # Replace with white
    img[mask] = [1.0, 1.0, 1.0]

    # Convert back to uint8
    result = (img * 255).astype(np.uint8)
    
    
    # cv2.imwrite(f"steps_out/{output_path}", result)
    return result



def improved_distortion(np_img: np.ndarray):
    """
    Improved distortion function that preserves digit features while adding realistic noise.
    
    Args:
        np_img: Input grayscale image (white background, black text).
    Returns:
        np.ndarray: Distorted grayscale image.
    """
    h, w = np_img.shape[:2]
    
    # Handle color images
    if len(np_img.shape) == 3:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    
    up = np_img.copy()
    # up = mask_pixels(up)
    # up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    
    
    # Step 1: Moderate downsampling (less aggressive than original)
    if random.random() < 0.6:
        scale = random.uniform(0.9, 0.95)  # Less aggressive scaling
        small = cv2.resize(up, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # # Step 2: Light noise addition
    # if random.random() < 0.4:
    #     noise = np.random.normal(0, random.uniform(2, 6), up.shape)
    #     up = np.clip(up.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Step 3: Slight blur (much less aggressive)
    # if random.random() < 0.5:
    # kernel_size = random.choice([3, 5])
    # sigma = random.uniform(0.2, 0.3)
    # up = cv2.GaussianBlur(up, (kernel_size, kernel_size), sigmaX=sigma)
    
    # Step 4: Moderate contrast adjustment
    # if random.random() < 0.4:
    #     alpha = random.uniform(0.9, 1.1)  # Less aggressive contrast
    #     beta = random.uniform(-5, 5)      # Less aggressive brightness
    #     up = np.clip(alpha * up.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    # # Step 5: Very light morphological operations
    # if random.random() < 0.2:
    #     kernel = np.ones((2, 2), np.uint8)
    #     if random.random() < 0.5:
    #         up = cv2.erode(up, kernel, iterations=1)
    #     else:
    #         up = cv2.dilate(up, kernel, iterations=1)
    
    # up = np.clip(up + 25,255, 0).astype(np.uint8)
    kernel_size = random.choice([1,3])
    
   
    
    up = cv2.GaussianBlur(up, (kernel_size, kernel_size), sigmaX=random.uniform(0.5, 0.6))
    
    up = mask_pixels(up)
    up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)


    
    return up

class ImprovedDigitGenerator:
    """Generate high-quality synthetic digit images with controlled distortion."""
    
    def __init__(self, fonts, img_size=32):
        self.fonts = fonts
        self.img_size = img_size
        self.digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    def create_digit_image(self, digit, font):
        """Create a 32x32 grayscale image with a centered digit."""
        # Create white background (255 = white)
        img = Image.new('L', (self.img_size, self.img_size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), digit, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center with small random offset
        x = (self.img_size - text_width) // 2 + random.randint(-1, 1)
        y = (self.img_size - text_height) // 2 + random.randint(-1, 1)
        
        # Draw black digit (0 = black)
        draw.text((x, y), digit, fill=20, font=font)
        
        return img
    
    def augment_digit(self, img):
        """Apply controlled augmentation to preserve digit features."""
        img_array = np.array(img)
        
        # Apply improved distortion (less aggressive)
        img_array = improved_distortion(img_array)
        
        return Image.fromarray(img_array, mode='L')
    
    def generate_dataset(self, samples_per_digit=800):
        """Generate complete dataset with controlled quality."""
        total_samples = samples_per_digit * 10
        print(f"Generating {total_samples} high-quality digit images...")
        print(f"  - {samples_per_digit} samples per digit (0-9)\n")
        
        X = []
        y = []
        
        for digit_idx, digit in enumerate(self.digits):
            print(f"  Generating digit '{digit}' ({digit_idx}/9)...", end=' ')
            
            for i in range(samples_per_digit):
                # Random font
                font = random.choice(self.fonts)
                
                # Create base image
                img = self.create_digit_image(digit, font)
                
                # Apply controlled augmentation
                if i < samples_per_digit // 3:
                    # First half: clean images
                    pass
                else:
                    # Second half: augmented images
                    img = self.augment_digit(img)
                
                # Convert to numpy and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                X.append(img_array)
                y.append(digit_idx)
            
            print(f"✓ ({len(X)} total)")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for CNN input
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        # Shuffle dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  - Total samples: {len(X)}")
        print(f"  - Shape: {X.shape}")
        print(f"  - Value range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  - Class distribution: {np.bincount(y)}\n")
        
        return X, y
    
    
    

def visualize_digit_samples(X, y, title="Sample Digits"):
    """
    Visualize one sample from each digit class (0-9).
    """
    print("Visualizing sample digits (one per class)...\n")
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Get one sample per digit
    for digit in range(10):
        idx = np.where(y == digit)[0][0]
        img = X[idx].reshape(32, 32)
        
        row = digit // 5
        col = digit % 5
        ax = axes[row, col]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Digit: {digit}", fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnt/numbers/digit_samples.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved!\n")
    plt.close()


def build_improved_cnn(input_shape=(32, 32, 1), num_classes=10):
    """Enhanced features:
        - 4 Convolutional blocks with residual-like connections
        - Better weight initialization
        - Improved regularization strategy
        - Multi-scale feature extraction
        - Better discrimination for similar digits (3/9, 5/6, 0/8)
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block - fine details
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    
    # Second convolutional block - medium features
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)
    
    
    
     # Third convolutional block - complex patterns
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    
    x = layers.Dropout(0.2)(x)

    # Fourth convolutional block - high-level features
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Enhanced dense layers with better regularization
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer with better initialization
    outputs = layers.Dense(num_classes, activation='softmax',
                          kernel_initializer='glorot_uniform')(x)
    
    
    model = models.Model(inputs=inputs, outputs=outputs, name='EnhancedDigitCNN')

    model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # print("Building improved CNN model...")
    
    # model = models.Sequential([
    #     # Input layer
    #     layers.Input(shape=input_shape),
        
    #     # First convolutional block
    #     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    #     layers.BatchNormalization(),
    #     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Dropout(0.1),
        
    #     # Second convolutional block
    #     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    #     layers.BatchNormalization(),
    #     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Dropout(0.1),
        
    #     # Third convolutional block
    #     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    #     layers.BatchNormalization(),
    #     layers.GlobalAveragePooling2D(),
        
    #     # Dense layers
    #     layers.Dense(256, activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.Dropout(0.5),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.3),
    #     layers.Dense(num_classes, activation='softmax')
    # ], name='ImprovedDigitCNN')
    
    # # Compile with optimized settings
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy']
    # )
    
    print("\nImproved Model Architecture:")
    print("=" * 70)
    model.summary()
    print("=" * 70)
    print()
    
    return model

import tensorflow as tf
import tensorflow.keras.backend as K



def create_confusion_aware_training_data(X, y, problematic_pairs=[(3,9), (5,6), (0,8)]):
    """
    Create additional training samples focused on problematic digit pairs.

    Args:
        X: Training images
        y: Training labels
        problematic_pairs: List of (digit1, digit2) pairs that get confused

    Returns:
        Enhanced X, y with additional samples for problematic digits
    """
    print("Creating confusion-aware training data...")

    enhanced_X = []
    enhanced_y = []

    # Add original data
    enhanced_X.extend(X)
    enhanced_y.extend(y)

    # Generate additional samples for problematic digits
    for digit1, digit2 in problematic_pairs:
        # Find samples of these digits
        digit1_indices = np.where(y == digit1)[0]
        digit2_indices = np.where(y == digit2)[0]

        # Add extra samples with slight variations
        for idx in digit1_indices[:50]:  # Take first 50 samples
            original_img = X[idx]

            # Create variations with different augmentations
            for _ in range(3):  # 3 variations per sample
                # Apply slight rotation and noise
                angle = np.random.uniform(-5, 5)
                noise = np.random.normal(0, 0.02, original_img.shape)

                # Simple rotation simulation (approximate)
                augmented = original_img + noise
                augmented = np.clip(augmented, 0, 1)

                enhanced_X.append(augmented)
                enhanced_y.append(digit1)
                
                for idx in digit2_indices[:50]:  # Take first 50 samples
                    original_img = X[idx]

                    # Create variations with different augmentations
                    for _ in range(3):  # 3 variations per sample
                        # Apply slight rotation and noise
                        angle = np.random.uniform(-5, 5)
                        noise = np.random.normal(0, 0.02, original_img.shape)

                        # Simple rotation simulation (approximate)
                        augmented = original_img + noise
                        augmented = np.clip(augmented, 0, 1)

                        enhanced_X.append(augmented)
                        enhanced_y.append(digit2)

    enhanced_X = np.array(enhanced_X)
    enhanced_y = np.array(enhanced_y)

    print(f"Enhanced dataset: {len(enhanced_X)} samples (was {len(X)})")
    return enhanced_X, enhanced_y


def enhanced_progressive_training(model, X_train, y_train, X_val, y_val,
                                 problematic_pairs=[(3,9), (5,6), (0,8)]):
    """
    Enhanced progressive training with focus on problematic digit pairs.

    Phase 1: Clean data training
    Phase 2: Confusion-aware training with problematic pairs
    Phase 3: Full augmentation training
    Phase 4: Fine-tuning with reduced learning rate
    """
    
    print("🚀 Starting Enhanced Progressive Training...")
    print("=" * 70)

    # Phase 1: Clean data training
    print("📚 Phase 1: Clean Data Training (Foundation)")
    print("-" * 50)

    # Use subset for initial training
    clean_indices = np.random.choice(len(X_train), size=min(4000, len(X_train)), replace=False)
    X_clean = X_train[clean_indices]
    y_clean = y_train[clean_indices]

    history1 = model.fit(
        X_clean, y_clean,
        validation_data=(X_val, y_val),
        epochs=8,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
        ],
        verbose=1
    )

    print(f"✓ Phase 1 complete. Best val accuracy: {max(history1.history['val_accuracy']):.4f}")

    # Phase 2: Confusion-aware training
    print("\n🎯 Phase 2: Confusion-Aware Training (Problematic Pairs)")
    print("-" * 50)

    # Create enhanced dataset with focus on problematic pairs
    X_enhanced, y_enhanced = create_confusion_aware_training_data(X_train, y_train, problematic_pairs)

    # Reduce learning rate for fine-tuning
    model.optimizer.learning_rate = 0.0001
    history2 = model.fit(
        X_enhanced, y_enhanced,
        validation_data=(X_val, y_val),
        epochs=6,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-7)
        ],
        verbose=1
    )

    print(f"✓ Phase 2 complete. Best val accuracy: {max(history2.history['val_accuracy']):.4f}")

    # Phase 3: Full training with augmentation
    print("\n🔄 Phase 3: Full Augmentation Training")
    print("-" * 50)

    # Create data generator with augmentation
    datagen = ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='constant',
        cval=0
    )
     # Further reduce learning rate
    model.optimizer.learning_rate = 0.00005

    history3 = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        steps_per_epoch=len(X_train) // 64,
        validation_data=(X_val, y_val),
        epochs=8,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-8)
        ],
        verbose=1
    )

    print(f"✓ Phase 3 complete. Best val accuracy: {max(history3.history['val_accuracy']):.4f}")

    # Phase 4: Final fine-tuning
    print("\n✨ Phase 4: Final Fine-tuning")
    print("-" * 50)
    
    # Very low learning rate for final polish
    model.optimizer.learning_rate = 0.00001

    history4 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,  # Smaller batch for more stable gradients
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
        verbose=1
    )

    print(f"✓ Phase 4 complete. Final val accuracy: {max(history4.history['val_accuracy']):.4f}")

    print("\n🎉 Enhanced Progressive Training Complete!")
    print("=" * 70)
    
    return model, [history1, history2, history3, history4]

    



def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for multi-class classification (e.g. digits 0–9).
    Works with integer y_true labels.
    """
    def focal_loss_fixed(y_true, y_pred):
        # Convert labels to one-hot if necessary
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])  # [batch, num_classes]

        # Clip predictions to prevent NaN in log
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Compute cross-entropy term
        cross_entropy = -y_true * K.log(y_pred)

        # Compute modulating factor
        weight = alpha * K.pow(1 - y_pred, gamma)

        # Compute focal loss
        loss = weight * cross_entropy

        # Return mean loss
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed


def progressive_training(model, X_train, y_train, X_val, y_val):
    """
    Progressive training strategy for better convergence.

    Phase 1: Train on clean data
    Phase 2: Fine-tune on mixed data
    Phase 3: Final training with full augmentation
    """
    print("Starting progressive training strategy...")

    # Phase 1: Clean data training
    print("\n=== PHASE 1: Clean Data Training ===")

    # Create clean dataset (first half of each class)
    clean_indices = []
    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0]
        clean_indices.extend(digit_indices[:len(digit_indices)//2])

    X_clean = X_train[clean_indices]
    y_clean = y_train[clean_indices]

    print(f"Clean training samples: {len(X_clean)}")

    # Train on clean data
    history1 = model.fit(
        X_clean, y_clean,
        batch_size=32,
        epochs=20,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Phase 2: Mixed data training
    print("\n=== PHASE 2: Mixed Data Training ===")

    # Light data augmentation
    light_augmentation = keras.Sequential([
        layers.RandomRotation(0.05, fill_mode='constant', fill_value=1.0),
        layers.RandomTranslation(0.05, 0.05, fill_mode='constant', fill_value=1.0),
    ])

    # Create augmented model
    augmented_model = models.Sequential([
        light_augmentation,
        model
    ])
    augmented_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower learning rate
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    # Train on full dataset with light augmentation
    history2 = augmented_model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        ],
        verbose=1
    )
    
    # combined_history = history2

    # Phase 3: Final fine-tuning
    print("\n=== PHASE 3: Final Fine-tuning ===")

    # Full augmentation
    full_augmentation = keras.Sequential([
        layers.RandomRotation(0.1, fill_mode='constant', fill_value=1.0),
        layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=1.0),
        layers.RandomZoom(0.1, fill_mode='constant', fill_value=1.0),
    ])

    # Create final model
    final_model = models.Sequential([
        full_augmentation,
        model
    ])
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Very low learning rate
        loss='sparse_categorical_crossentropy',
        # loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),

        metrics=['accuracy']
    )

    # Final training
    history3 = final_model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ],
        verbose=1
    )

    print("\n✓ Progressive training complete!")

    # Combine histories
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'] + history3.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'] + history3.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'] + history3.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'] + history3.history['val_loss']
    }

    return model, combined_history

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation."""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✓ Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    for digit in range(10):
        digit_mask = y_test == digit
        if np.sum(digit_mask) > 0:
            digit_acc = np.mean(y_pred[digit_mask] == y_test[digit_mask])
            print(f"  Digit {digit}: {digit_acc:.4f} ({digit_acc*100:.1f}%)")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📈 Confusion Matrix:")
    print("    Predicted:")
    print("    ", "  ".join(f"{i:3d}" for i in range(10)))
    for i, row in enumerate(cm):
        print(f" {i}: ", "  ".join(f"{val:3d}" for val in row))

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

    return y_pred, y_pred_proba

def main():
    """Main execution pipeline with improved training."""
    # Create output directory
    os.makedirs('models/improved_results', exist_ok=True)

    # Step 1: Load fonts
    fonts = load_system_fonts()

    # Step 2: Generate high-quality dataset
    generator = ImprovedDigitGenerator(fonts, img_size=32)
    X, y = generator.generate_dataset(samples_per_digit=1000)
    
    visualize_digit_samples(X, y, title="Generated Digit Samples (One Per Class)")


    # Step 3: Split dataset
    print("Splitting dataset into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = X_train.astype("float32") / 255.0
    X_val   = X_val.astype("float32") / 255.0
    # X_test  = X_test.astype("float32") / 255.0
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Validation set: {len(X_val)} samples\n")
    
    import random
    import matplotlib.pyplot as plt

    samples = random.sample(range(len(X_train)), 25)
    # fig, axs = plt.subplots(5, 5, figsize=(6, 6))
    # for ax, i in zip(axs.ravel(), samples):
    #     ax.imshow(X_train[i].squeeze(), cmap='gray')
    #     ax.set_title(int(y_train[i]))
    #     ax.axis('off')
    # plt.show()

    # Step 4: Build improved model
    model = build_improved_cnn(input_shape=(32, 32, 1), num_classes=10)
    # 2️⃣ Check image range
    print("X_train range:", X_train.min(), X_train.max())
    print("Mean:", np.mean(X_train), "Std:", np.std(X_train))

    # 3️⃣ Check model predictions before training
    preds = model.predict(X_train[:50], verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    print("Unique preds before training:", np.unique(pred_classes, return_counts=True))

    # 4️⃣ Check output probabilities
    print("Sample logits (first 5 predictions):")
    print(preds[:5])

    # Optional: visualize a few samples and their random predictions
    # import matplotlib.pyplot as plt
    # for i in range(5):
    #     plt.imshow(X_train[i].squeeze(), cmap='gray')
    #     plt.title(f"Pred before training: {pred_classes[i]}")
    #     plt.show()
    preds = model.predict(X_train[:50])
    print("Unique preds before training:", np.unique(np.argmax(preds, axis=1)))
    print("y_train shape:", y_train.shape)
    print("Example:", y_train[0])
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))
    print(model.output_shape)
    
    print("Initial loss (on small batch):",
    model.evaluate(X_train[:32], y_train[:32], verbose=0))

    print("X_train:", X_train.shape, X_train.min(), X_train.max())
    print("y_train:", y_train.shape, "unique labels:", np.unique(y_train))
    print("X_val:", X_val.shape, X_val.min(), X_val.max())


    # Step 5: Progressive training
    model, history = progressive_training(model, X_train, y_train, X_val, y_val)


    # Step 5: Enhanced Progressive Training with focus on problematic pairs
    print("\n🎯 Starting Enhanced Training for Better Digit Discrimination...")
    problematic_pairs = [(3, 9), (5, 6), (0, 8)]  # Your identified problem pairs

    # model, histories = enhanced_progressive_training(
    #     model, X_train, y_train, X_val, y_val,
    #     problematic_pairs=problematic_pairs
    # )
    
    # Step 6: Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_val, y_val)

    # Step 7: Save model
    model_path = 'models/try_one.keras'
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Step 8: Save training history
    np.save('models/improved_results/training_history.npy', history)

    print("\n" + "=" * 70)
    print("🎉 IMPROVED TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📈 Expected improvements:")
    print("  - Accuracy should be >95% (vs previous 11%)")
    print("  - Better generalization to distorted images")
    print("  - More stable training convergence")
    print("  - Reduced overfitting")

    return model, history

if __name__ == "__main__":
    # model, history = main()
    main()

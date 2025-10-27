#!/usr/bin/env python3
"""
Binary CNN Classifier: Number Detection in 32x32 Grayscale Patches
===================================================================
This script generates synthetic image patches, trains a CNN to classify them
as either NUMBER or OTHER, and evaluates the model's performance.

Classes:
    - NUMBER (label=1): digits, multi-digit numbers, decimals
    - OTHER (label=0): letters, symbols, punctuation, blank, noise
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
import os

print("=" * 70)
print("NUMBER vs OTHER CNN CLASSIFIER")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print()


# ============================================================================
# SECTION 1: FONT LOADING
# ============================================================================

def load_system_fonts():
    """
    Attempt to load common system fonts. Falls back to default if unavailable.
    Returns a list of ImageFont objects.
    """
    """
    Attempt to load common system fonts. Falls back to default if unavailable.
    Returns a list of ImageFont objects.
    """
    fonts = []
    font_names = [
        "Times New Roman.ttf", "Arial.ttf", "Arial Bold.ttf","Times.ttf",
       
    ]

    font_sizes = [22, 24]

    # Common macOS + cross-platform font directories
    font_paths = [
        "/System/Library/Fonts/",
        "/Library/Fonts/",
        os.path.expanduser("~/Library/Fonts/"),
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
        "/usr/share/fonts/",
        "C:\\Windows\\Fonts\\",  # keep for Windows compatibility
    ]

    loaded_count = 0
    for size in font_sizes:
        for font_name in font_names:
            for path in font_paths:
                font_file = os.path.join(path, font_name)
                if os.path.exists(font_file):
                    try:
                        font = ImageFont.truetype(font_file, size)
                        fonts.append(font)
                        loaded_count += 1
                        print(f"Loaded: {font_name} ({size}px)")
                        break
                    except Exception as e:
                        print(f"Failed to load {font_file}: {e}")
                        continue
            if loaded_count >= 10:  # Limit to 10 loaded fonts
                break
        if loaded_count >= 10:
            break
    fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/calibri-regular.ttf", 12))
    fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/Lucida Console Regular.ttf", 12))
    
        
    print(f"Loaded {len(fonts)} font variations\n")
    return fonts




def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
        """
        Simulate low-res, pixelated distortion for white text on black background.

        Args:
            np_img: Input grayscale image (white text on black background).
            min_scale, max_scale: Downsampling scale range.
        Returns:
            np.ndarray: Distorted grayscale image.
        """
        import cv2
        
        max_shift = 1
        blur_chance = 0.5

        h, w = np_img.shape[:2]

        # Step 1: Randomly scale down & up (pixelation)
        scale = random.uniform(0.9, 0.95)
        small = cv2.resize(np_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # lighther pixels a bit
        # up = up.astype(np.float32)
        # factor = random.uniform(0.2, 0.3)  # how much to dim bright areas
        # mask = up > 155  # only bright pixels (the text)
        # up[mask] = up[mask] - (up[mask] - 155) * factor
        # up = np.clip(up, 0, 255).astype(np.uint8)

        # # Step 2: Apply small random affine transform (to jitter or shift)
        # dx = random.randint(-max_shift, max_shift)
        # dy = random.randint(-max_shift, max_shift)
        # M = np.float32([[1, 0, dx], [0, 1, dy]])
        # up = cv2.warpAffine(up, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # # Step 3: Lightly “fade” bright regions (since text is white)
        # up = up.astype(np.float32)
        # factor = random.uniform(0.2, 0.3)  # how much to dim bright areas
        # mask = up > 155  # only bright pixels (the text)
        # up[mask] = up[mask] - (up[mask] - 155) * factor  # pull toward gray
        # up = np.clip(up, 0, 255).astype(np.uint8)

        # # Step 4: Optional small blur to soften edges
        
        up = cv2.GaussianBlur(up, (3, 3), sigmaX=random.uniform(0.3, 0.4))

        return up
# ============================================================================
# SECTION 2: DATASET GENERATION
# ============================================================================

class SyntheticDataGenerator:
    """
    Generates synthetic 32x32 grayscale image patches for binary classification.
    """
    
    def __init__(self, fonts, img_size=32):
        self.fonts = fonts
        self.img_size = img_size
        
        self.number_tokens = [
            # Single digits
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Two-digit numbers
            '10', '11', '12', '20', '25', '32', '48', '64', '77', '99',
            "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
            "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
            "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
            "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
            "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
            "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
            
            # Decimals
            '.1', '.5', '.9', '0.2', '0.5', '1.5', '2.0', '3.14', '20.',
            # Edge cases
            '00', '01', '100',
        ]
        
        self.other_tokens = [
            # Single uppercase Cyrillic letters
            'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М',
            'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ы', 'Э', 'Ю', 'Я',

            # Single lowercase Cyrillic letters
            'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м',
            'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ы', 'э', 'ю', 'я',

            # Common Cyrillic letter combinations (units, abbreviations, etc.)
            'см', 'мм', 'кг', 'гр', 'час', 'руб', 'коп', 'шт', 'мл', 'л', 'тг', 'мг',
            'Москва', 'СПБ', 'ООО', 'ТД', 'Цена', 'Сум', 'грн', 'BYN',
            # Symbols
            '$', '€', '£', '%', '°', '/', ':', '@', '#', '&', '*',
            '+', '-', '=', '~', '_', '|', '\\', '<', '>',
            # Punctuation
            '.', ',', ';', '!', '?', "'", '"', '(', ')', '[', ']', '{', '}',
            # Empty string for blank patches
            '',
        ]
    
    def create_text_image(self, text, font):
        """
        Create a 32x32 grayscale image with centered text.
        """
        # Create white background
        img = Image.new('L', (self.img_size, self.img_size), color=255)
        draw = ImageDraw.Draw(img)
        
        if text:  # Only draw if text is not empty
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text
            x = (self.img_size - text_width) // 2
            y = (self.img_size - text_height) // 2
            
            # Draw black text
            draw.text((x, y), text, fill=0, font=font)
        
        return img
    
    def create_noise_image(self):
        """
        Create a random noise patch.
        """
        noise_type = random.choice(['gaussian', 'salt_pepper', 'blur'])
        
        if noise_type == 'gaussian':
            # Gaussian noise
            noise = np.random.randint(128, 255, (self.img_size, self.img_size), dtype=np.uint8)
            img = Image.fromarray(noise, mode='L')
        
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise
            img = Image.new('L', (self.img_size, self.img_size), color=255)
            pixels = img.load()
            for i in range(self.img_size):
                for j in range(self.img_size):
                    if random.random() < 0.1:
                        pixels[i, j] = random.choice([0, 255])
        
        else:  # blur
            # Create some random shapes and blur
            img = Image.new('L', (self.img_size, self.img_size), color=255)
            draw = ImageDraw.Draw(img)
            for _ in range(5):
                x1, y1 = random.randint(0, self.img_size), random.randint(0, self.img_size)
                x2, y2 = random.randint(0, self.img_size), random.randint(0, self.img_size)
                draw.line([x1, y1, x2, y2], fill=random.randint(0, 128), width=2)
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        return img
    
    
    

    
    def augment_image(self, img):
        """
        Apply random augmentations to an image.
        """
        
        # # Random rotation (±6 degrees)
        # if random.random() < 0.5:
        #     angle = random.uniform(-6, 6)
        #     img = img.rotate(angle, fillcolor=255)
        
        # # Random translation (up to 10%)
        # if random.random() < 0.5:
        #     max_shift = int(0.1 * self.img_size)
        #     dx = random.randint(-max_shift, max_shift)
        #     dy = random.randint(-max_shift, max_shift)
        #     img = Image.new('L', (self.img_size, self.img_size), color=255)
        #     temp = img.copy()
        #     img.paste(temp, (dx, dy))
        
        # # Random scaling (±10%)
        # if random.random() < 0.3:
        #     scale = random.uniform(0.9, 1.1)
        #     new_size = int(self.img_size * scale)
        #     img = img.resize((new_size, new_size), Image.LANCZOS)
        #     # Crop or pad to original size
        #     if new_size > self.img_size:
        #         left = (new_size - self.img_size) // 2
        #         top = (new_size - self.img_size) // 2
        #         img = img.crop((left, top, left + self.img_size, top + self.img_size))
        #     else:
        #         new_img = Image.new('L', (self.img_size, self.img_size), color=255)
        #         offset = (self.img_size - new_size) // 2
        #         new_img.paste(img, (offset, offset))
        #         img = new_img
        
        # Add Gaussian noise
        # if random.random() < 0.3:
        #     img_array = np.array(img)
        #     noise = np.random.normal(0, 10, img_array.shape)
        #     img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        #     img = Image.fromarray(img_array, mode='L')
        
        # # Random contrast
        # if random.random() < 0.3:
        #     from PIL import ImageEnhance
        #     enhancer = ImageEnhance.Contrast(img)
        #     img = enhancer.enhance(random.uniform(0.8, 1.2))
            
        np_img = np.array(img).astype(np.float32) / 255.0
        np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)
        img = Image.fromarray((np_img * 255).astype(np.uint8), mode='L')
        
        return img
    
    def generate_dataset(self, n_samples=4000, augment=True):
        """
        Generate a balanced dataset of NUMBER and OTHER samples.
        
        Args:
            n_samples: Total number of samples to generate
            augment: Whether to apply augmentations
        
        Returns:
            X: numpy array of shape (n_samples, 32, 32, 1)
            y: numpy array of shape (n_samples,) with labels
        """
        print(f"Generating {n_samples} synthetic images...")
        
        X = []
        y = []
        
        samples_per_class = n_samples // 2
        
        # Generate NUMBER class (label=1)
        print(f"  - Generating {samples_per_class} NUMBER samples...")
        for i in range(samples_per_class):
            token = random.choice(self.number_tokens)
            font = random.choice(self.fonts)
            img = self.create_text_image(token, font)
            
            if augment:
                img = self.augment_image(img)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            X.append(img_array)
            y.append(1)
        
        # Generate OTHER class (label=0)
        print(f"  - Generating {samples_per_class} OTHER samples...")
        for i in range(samples_per_class):
            # 10% chance of noise image
            if random.random() < 0.1:
                img = self.create_noise_image()
            else:
                token = random.choice(self.other_tokens)
                font = random.choice(self.fonts)
                img = self.create_text_image(token, font)
            
            if augment:
                img = self.augment_image(img)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            X.append(img_array)
            y.append(0)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to (n_samples, 32, 32, 1) for CNN input
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"  - Dataset shape: {X.shape}")
        print(f"  - Labels shape: {y.shape}")
        print(f"  - Class distribution: {np.sum(y == 0)} OTHER, {np.sum(y == 1)} NUMBER\n")
        
        return X, y


# ============================================================================
# SECTION 3: VISUALIZATION
# ============================================================================

def visualize_samples(X, y, n_samples=16, title="Sample Images"):
    """
    Visualize a grid of sample images from the dataset.
    """
    print(f"Visualizing {n_samples} random samples...\n")
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        img_idx = indices[idx]
        img = X[img_idx].reshape(32, 32)
        label = y[img_idx]
        label_text = "NUMBER" if label == 1 else "OTHER"
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"{label_text}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnt/outputs/sample_images.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved!\n")
    plt.close()


# ============================================================================
# SECTION 4: MODEL DEFINITION
# ============================================================================

def build_cnn_model(input_shape=(32, 32, 1)):
    """
    Build a simple CNN for binary classification of 32x32 grayscale images.
    
    Architecture:
        - Conv2D (32 filters) + MaxPooling
        - Conv2D (64 filters) + MaxPooling
        - GlobalAveragePooling2D
        - Dense (64) + ReLU
        - Dense (1) + Sigmoid
    """
    print("Building CNN model...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        # First convolutional block
        layers.Conv2D(32, (3,3), activation='relu', padding='same',),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(name='gap'),
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(1, activation='sigmoid', name='output')
    ], name='NumberDetectorCNN')
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    print("=" * 70)
    model.summary()
    print("=" * 70)
    print()
    
    return model


# ============================================================================
# SECTION 5: TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=25, batch_size=64):
    """
    Train the CNN model with data augmentation.
    """
    print(f"Starting training for {epochs} epochs...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}\n")
    
    # Data augmentation layers (applied during training)
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.05, fill_mode='constant', fill_value=1.0),
        layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=1.0),
    ])
    
    # Create augmented model
    augmented_model = models.Sequential([
        data_augmentation,
        model
    ])
    
    # ✅ Re-compile the augmented model
    augmented_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    
    # Train the model
    history = augmented_model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\nTraining complete!\n")
    
    return history


# ============================================================================
# SECTION 6: EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and display metrics.
    """
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\nValidation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              OTHER  NUMBER")
    print(f"Actual OTHER  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"       NUMBER {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['OTHER', 'NUMBER'],
        digits=4
    ))
    
    return y_pred, y_pred_proba


def visualize_predictions(model, X_test, y_test, n_samples=10):
    """
    Visualize predictions on random test samples.
    """
    print(f"Visualizing {n_samples} random predictions...\n")
    
    # Select random samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Model Predictions on Test Set', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx >= n_samples:
            break
        
        img_idx = indices[idx]
        img = X_test[img_idx].reshape(32, 32)
        true_label = y_test[img_idx]
        
        # Get prediction
        pred_proba = model.predict(X_test[img_idx:img_idx+1], verbose=0)[0][0]
        pred_label = 1 if pred_proba > 0.5 else 0
        
        # Determine label text and color
        true_text = "NUMBER" if true_label == 1 else "OTHER"
        pred_text = "NUMBER" if pred_label == 1 else "OTHER"
        confidence = pred_proba if pred_label == 1 else (1 - pred_proba)
        
        color = 'green' if pred_label == true_label else 'red'
        
        # Display image
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"True: {true_text}\nPred: {pred_text}\nConf: {confidence:.2%}",
            fontsize=9,
            color=color,
            fontweight='bold'
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnt/outputs/predictions.png', dpi=150, bbox_inches='tight')
    print("Prediction visualization saved!\n")
    plt.close()


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    print("Plotting training history...\n")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnt/outputs/training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved!\n")
    plt.close()


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    # Step 1: Load fonts
    fonts = load_system_fonts()
    
    # # Step 2: Generate dataset
    # generator = SyntheticDataGenerator(fonts, img_size=32)
    # X, y = generator.generate_dataset(n_samples=4000, augment=True)
    
    # # Step 3: Visualize samples
    # visualize_samples(X, y, n_samples=16, title="Generated Dataset Samples")
    
    # # Step 4: Split dataset
    # print("Splitting dataset into train/validation sets...")
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
    
    
    
     # Step 2: Generate two *separate* datasets
    generator = SyntheticDataGenerator(fonts, img_size=32)

    print("Generating training and validation datasets separately...")

    # TRAINING SET — with augmentation and more variation
    X_train, y_train = generator.generate_dataset(n_samples=20000, augment=True)
    
    visualize_samples(X_train, y_train, n_samples=16, title="Generated Dataset Samples")


    print(X_train.min(), X_train.max(), X_train.mean())


    # VALIDATION SET — without augmentation (cleaner, independent)
    X_val, y_val = generator.generate_dataset(n_samples=2000, augment=True)
    visualize_samples(X_val, y_val, n_samples=16, title="Validation Samples (Clean)")


    
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Validation set: {len(X_val)} samples\n")
    
    # Step 5: Build model
    model = build_cnn_model(input_shape=(32, 32, 1))
    
    # Step 6: Train model
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=25, batch_size=64
    )
    
    # Step 7: Plot training history
    plot_training_history(history)
    
    # Step 8: Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_val, y_val)
    
    # Model evaluation is:
    print("Model evaluation is:")
    print(classification_report(
        y_val, y_pred,
        target_names=['OTHER', 'NUMBER'],
        digits=4
    ))
    
    # Step 9: Visualize predictions
    visualize_predictions(model, X_val, y_val, n_samples=10)
    
    # Step 10: Save model
    model_path = 'mnt/outputs/number_detector_cnn.keras'
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. sample_images.png - Dataset sample visualization")
    print("  2. training_history.png - Training curves")
    print("  3. predictions.png - Model predictions visualization")
    print("  4. number_detector_cnn.h5 - Trained model weights")
    print("\n" + "=" * 70)
    
    # Optional: Integration suggestion
    print("\nOPTIONAL EXTENSION - Two-Stage OCR Pipeline:")
    print("-" * 70)
    print("""
This CNN classifier can be integrated into a two-stage OCR pipeline:

STAGE 1: Number Detection (This Model)
    - Input: 32x32 grayscale image patch
    - Output: Binary classification (NUMBER or OTHER)
    - Purpose: Filter out non-numeric content before OCR
    - Benefit: Reduces false positives in numeric extraction

STAGE 2: Number Recognition (CRNN + CTC)
    - Input: Patches classified as NUMBER from Stage 1
    - Architecture: CNN layers + RNN (LSTM/GRU) + CTC loss
    - Output: Sequence of digits (e.g., "123", "45.67")
    - Purpose: Read the actual numeric value

Pipeline Flow:
    Image Patch → Stage 1 (This CNN) → [if NUMBER] → Stage 2 (CRNN) → Final Text
                                     → [if OTHER] → Discard or different pipeline

Benefits:
    - Improved accuracy by pre-filtering
    - Faster processing (skip OCR for non-numbers)
    - Can use different models for numbers vs text
    - Better handling of mixed content documents
    """)
    print("=" * 70)


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()
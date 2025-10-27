#!/usr/bin/env python3
"""
CNN Digit Classifier: Synthetic Handwritten-Style Digits 0-9
==============================================================
This script generates synthetic digit images and trains a CNN to classify
handwritten-style digits (0-9) from 32x32 grayscale patches.

Output: 10-class classifier (one per digit)
Training data: Fully synthetic, no external datasets required
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import random
import warnings
import os
import cv2
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

print("=" * 70)
print("HANDWRITTEN-STYLE DIGIT CNN CLASSIFIER (0-9)")
print("=" * 70)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print()


# ============================================================================
# SECTION 1: FONT LOADING
# ============================================================================

def load_system_fonts():
    """
    Load available system fonts for digit rendering.
    Falls back to default font if no TrueType fonts found.
    """
    print("Loading system fonts...")
    font_size = 12
    fonts = []
    # Try to use a simple font
    font = ImageFont.truetype("/Users/mariaivanova/Library/Fonts/calibri-regular.ttf", font_size)
    fonts.append(font)
    fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/Lucida Console Regular.ttf", 14))
    fonts.append(font)
    font=ImageFont.truetype("/Library/Fonts/adelle-sans-regular.otf", font_size)
    fonts.append(font)
    font_sizes = [12, 14, 16, 18, 20, 22]
    
    # Common font paths across different OS
    # font_paths = [
    #     "/usr/share/fonts/truetype/dejavu/",
    #     "/usr/share/fonts/truetype/liberation/",
    #     "/usr/share/fonts/truetype/liberation2/",
    #     "/usr/share/fonts/",
    #     "/System/Library/Fonts/",
    #     "C:\\Windows\\Fonts\\",
    #     "/usr/share/fonts/truetype/",
    # ]
    
    # loaded_count = 0
    # for size in font_sizes:
    #     for font_name in font_names:
    #         for path in font_paths:
    #             try:
    #                 font_file = os.path.join(path, font_name)
    #                 if os.path.exists(font_file):
    #                     font = ImageFont.truetype(font_file, size)
    #                     fonts.append(font)
    #                     loaded_count += 1
    #                     if loaded_count >= 1:  # Found at least one
    #                         break
    #             except:
    #                 continue
    #         if loaded_count >= 15:  # Limit total fonts
    #             break
    #     if loaded_count >= 15:
    #         break
    
    # # Fallback to default font
    # if len(fonts) == 0:
    #     print("Warning: No TrueType fonts found. Using default font.")
    #     for size in [12, 14, 16, 18, 20]:
    #         fonts.append(ImageFont.load_default())
    
    print(f"Loaded {len(fonts)} font variations\n")
    return fonts


# ============================================================================
# SECTION 2: SYNTHETIC DIGIT GENERATION
# ============================================================================
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
    bright_thresh = 0.87
    color_thresh = 0.1

    # Mask for light gray pixels
    mask = (brightness > bright_thresh) & (color_diff < color_thresh)

    # Replace with white
    img[mask] = [1.0, 1.0, 1.0]

    # Convert back to uint8
    result = (img * 255).astype(np.uint8)
    
    
    # cv2.imwrite(f"steps_out/{output_path}", result)
    return result


def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    """
    Simulate low-res, pixelated distortion for white text on black background.

    Args:
        np_img: Input grayscale image (white text on black background).
        min_scale, max_scale: Downsampling scale range.
    Returns:
        np.ndarray: Distorted grayscale image.
    """
    max_shift = 3
    blur_chance = 0.5

    h, w = np_img.shape[:2]

    # Step 1: Randomly scale down & up (pixelation)
    # scale = random.uniform(0.8, 0.83)
    # small = cv2.resize(np_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    # up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    up = np_img
    up = mask_pixels(up)
    up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    # Set a threshold — e.g., remove all pixels darker than 50
    threshold_value = 50


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12,12))
    up = clahe.apply(up)
    up = mask_pixels(up)
    up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    # Step 2: Apply small random affine transform (to jitter or shift)
    # dx = random.randint(-max_shift, max_shift)
    # dy = random.randint(-max_shift, max_shift)
    # M = np.float32([[1, 0, dx], [0, 1, dy]])
    # up = cv2.warpAffine(up, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # if random.random() < blur_chance:
        
    #     # Step 3: Lightly “fade” bright regions (since text is white)
    up = up.astype(np.float32)
    factor = random.uniform(0.5, 0.8)  # how much to dim bright areas
    mask = up > 155  # only bright pixels (the text)
    up[mask] = up[mask] - (up[mask] - 155) * factor  # pull toward gray
    up = np.clip(up, 0, 255).astype(np.uint8)
    # lighten_chance = 1.0  # always apply (or set <1.0 for random)
    # if random.random() < lighten_chance:
    #     up = up.astype(np.float32)
    #     factor = random.uniform(0.9, 1.0)  # how much to lighten dark regions
    #     dark_mask = up < 20                # identify text/dark regions
    #     up[dark_mask] = up[dark_mask] + (255 - up[dark_mask]) * factor
    #     up = np.clip(up, 0, 255).astype(np.uint8)

    # Step 4: Optional small blur to soften edges
    
    _, img = cv2.threshold(up, threshold_value, 255, cv2.THRESH_TOZERO)


    kernel_size = random.choice([3, 5, 7])

    
    up = cv2.GaussianBlur(up, (kernel_size, kernel_size), sigmaX=random.uniform(0.7, 1.0))

    return up



class SyntheticDigitGenerator:
    """
    Generate synthetic handwritten-style digit images (0-9).
    """
    
    def __init__(self, fonts, img_size=32):
        self.fonts = fonts
        self.img_size = img_size
        self.digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    def create_digit_image(self, digit, font):
        """
        Create a 32x32 grayscale image with a centered digit.
        
        Args:
            digit: String digit ('0'-'9')
            font: ImageFont object
        
        Returns:
            PIL Image (grayscale)
        """
        # Create white background
        img = Image.new('L', (self.img_size, self.img_size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), digit, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center with small random offset for variation
        x = (self.img_size - text_width) // 2 + random.randint(-2, 2)
        y = (self.img_size - text_height) // 2 + random.randint(-2, 2)
        
        # Draw digit with variable darkness (simulating ink variation)
        text_darkness = random.randint(200, 220)  # 0=pure black, higher=lighter
        draw.text((x, y), digit, font=font)
        
        # # Occasional ghosting/double print effect
        # if random.random() < 0.1:
        #     ghost_offset = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        #     draw.text((x + ghost_offset[0], y + ghost_offset[1]), digit,
        #              fill=min(text_darkness + 120, 200), font=font)
        
        return img
    
    
    def augment_digit(self, img):
        
        img_array = np.array(img)
        print("TYPE: ", img_array.dtype)
        img_array = downsample_then_upsample(img_array)
        # img = Image.fromarray(img_array, mode='L')
        # """
        # Apply realistic augmentations to simulate handwritten variation.
        
        # Args:
        #     img: PIL Image
        
        # Returns:
        #     Augmented PIL Image
        # """
        # import cv2
        
        # # Convert to numpy
        # img_array = np.array(img)
        # h, w = img_array.shape
        
        # # 1. Random rotation (±15 degrees)
        # if random.random() < 0.8:
        #     angle = random.uniform(-15, 15)
        #     M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        #     img_array = cv2.warpAffine(img_array, M, (w, h), 
        #                                borderMode=cv2.BORDER_REPLICATE,
        #                                borderValue=255)
        
        # # 2. Random translation (±10%)
        # if random.random() < 0.7:
        #     dx = random.randint(-3, 3)
        #     dy = random.randint(-3, 3)
        #     M = np.float32([[1, 0, dx], [0, 1, dy]])
        #     img_array = cv2.warpAffine(img_array, M, (w, h),
        #                                borderMode=cv2.BORDER_REPLICATE,
        #                                borderValue=255)
        
        # # 3. Random scaling (0.8-1.2x)
        # if random.random() < 0.6:
        #     scale = random.uniform(0.8, 1.2)
        #     scaled = cv2.resize(img_array, (int(w*scale), int(h*scale)),
        #                        interpolation=cv2.INTER_LINEAR)
            
        #     # Crop or pad to original size
        #     if scale > 1.0:
        #         # Crop from center
        #         start_x = (scaled.shape[1] - w) // 2
        #         start_y = (scaled.shape[0] - h) // 2
        #         img_array = scaled[start_y:start_y+h, start_x:start_x+w]
        #     else:
        #         # Pad with white
        #         pad_x = (w - scaled.shape[1]) // 2
        #         pad_y = (h - scaled.shape[0]) // 2
        #         img_array = np.full((h, w), 255, dtype=np.uint8)
        #         img_array[pad_y:pad_y+scaled.shape[0], 
        #                  pad_x:pad_x+scaled.shape[1]] = scaled
        
        # # 4. Random shear/perspective (subtle)
        # if random.random() < 0.3:
        #     pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        #     shear = random.uniform(-0.1, 0.1)
        #     pts2 = np.float32([
        #         [0, shear*h], [w, 0], 
        #         [0, h], [w, h-shear*h]
        #     ])
        #     M = cv2.getPerspectiveTransform(pts1, pts2)
        #     img_array = cv2.warpPerspective(img_array, M, (w, h),
        #                                    borderMode=cv2.BORDER_REPLICATE,
        #                                    borderValue=255)
        
        # # 5. Brightness/Contrast variation
        # if random.random() < 0.7:
        #     alpha = random.uniform(0.7, 1.3)  # Contrast
        #     beta = random.uniform(-30, 30)    # Brightness
        #     img_array = np.clip(alpha * img_array + beta, 0, 255).astype(np.uint8)
        
        # # 6. Gaussian blur (simulate pen stroke blur)
        # if random.random() < 0.5:
        #     kernel_size = random.choice([3, 5])
        #     img_array = cv2.GaussianBlur(img_array, (kernel_size, kernel_size),
        #                                 sigmaX=random.uniform(0.3, 1.0))
        
        # # 7. Gaussian noise
        # if random.random() < 0.4:
        #     noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
        #     img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # # 8. Morphological operations (erosion/dilation)
        # if random.random() < 0.25:
        #     kernel = np.ones((2, 2), np.uint8)
        #     if random.random() < 0.5:
        #         img_array = cv2.erode(img_array, kernel, iterations=1)
        #     else:
        #         img_array = cv2.dilate(img_array, kernel, iterations=1)
        
        # # 9. Salt and pepper noise
        # if random.random() < 0.2:
        #     prob = random.uniform(0.005, 0.02)
        #     rnd = np.random.random(img_array.shape)
        #     img_array[rnd < prob/2] = 0
        #     img_array[rnd > 1 - prob/2] = 255
        
        # # 10. Elastic distortion (simulate pen pressure variation)
        # if random.random() < 0.15:
        #     img_array = self._elastic_transform(img_array, 
        #                                         alpha=random.uniform(2, 8),
        #                                         sigma=random.uniform(2, 4))
        
        return Image.fromarray(img_array, mode='L')
    
    def _elastic_transform(self, image, alpha, sigma):
        """
        Elastic deformation of images for more natural variation.
        """
        import cv2
        from scipy.ndimage import gaussian_filter
        
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
        
        return cv2.remap(image, indices[1], indices[0], 
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    
    def generate_dataset(self, samples_per_digit=600):
        """
        Generate complete dataset of synthetic digits.
        
        Args:
            samples_per_digit: Number of samples per digit class
        
        Returns:
            X: numpy array of shape (n_samples, 32, 32, 1)
            y: numpy array of shape (n_samples,) with digit labels (0-9)
        """
        total_samples = samples_per_digit * 10
        print(f"Generating {total_samples} synthetic digit images...")
        print(f"  - {samples_per_digit} samples per digit (0-9)\n")
        
        X = []
        y = []
        
        for digit_idx, digit in enumerate(self.digits):
            print(f"  Generating digit '{digit}' ({digit_idx}/9)...", end=' ')
            
            for _ in range(samples_per_digit):
                # Random font
                font = random.choice(self.fonts)
                
                # Create base image
                img = self.create_digit_image(digit, font)
                
                # Apply augmentation
                img = self.augment_digit(img)
                
                # Convert to numpy and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                X.append(img_array)
                y.append(digit_idx)
            
            print(f"✓ ({len(X)} total)")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X for CNN input
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        # Shuffle dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  - Total samples: {len(X)}")
        print(f"  - Shape: {X.shape}")
        print(f"  - Labels shape: {y.shape}")
        print(f"  - Class distribution: {np.bincount(y)}\n")
        
        return X, y


# ============================================================================
# SECTION 3: VISUALIZATION
# ============================================================================

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


def visualize_random_samples(X, y, n_samples=20, title="Random Samples"):
    """
    Visualize random samples from the dataset.
    """
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    for idx, ax in enumerate(axes.flat):
        img_idx = indices[idx]
        img = X[img_idx].reshape(32, 32)
        label = y[img_idx]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {label}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnt/numbers/random_digit_samples.png', dpi=150, bbox_inches='tight')
    print("Random samples visualization saved!\n")
    plt.close()

 
# ============================================================================
# SECTION 4: MODEL DEFINITION
# ============================================================================

def build_digit_cnn(input_shape=(32, 32, 1), num_classes=10):
    """
    Build a compact CNN for digit classification (0-9).
    
    Architecture:
        - Conv2D (32 filters) + MaxPooling
        - Conv2D (64 filters) + MaxPooling
        - GlobalAveragePooling2D
        - Dense (128) + ReLU
        - Dense (10) + Softmax
    
    Returns:
        Compiled Keras model
    """
    print("Building CNN model...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(name='gap'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='DigitClassifierCNN')
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
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

def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=64):
    """
    Train the CNN model with data augmentation.
    """
    print(f"Starting training for {epochs} epochs...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {batch_size}\n")
    
    # Data augmentation layers
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.08, fill_mode='constant', fill_value=1.0),
        layers.RandomTranslation(0.1, 0.1, fill_mode='constant', fill_value=1.0),
        layers.RandomZoom(0.1, fill_mode='constant', fill_value=1.0),
    ])
    
    # Augmented model
    augmented_model = models.Sequential([
        data_augmentation,
        model
    ])
    augmented_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # or categorical_crossentropy (see below)
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
    
    # Train
    history = augmented_model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("\n✓ Training complete!\n")
    
    return history


# ============================================================================
# SECTION 6: EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and display metrics.
    """
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"\n✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Confusion Matrix:")
    print("    Predicted:")
    print("    ", "  ".join(f"{i:3d}" for i in range(10)))
    for i, row in enumerate(cm):
        print(f" {i}: ", "  ".join(f"{val:3d}" for val in row))
    
    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    
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
        pred_proba = model.predict(X_test[img_idx:img_idx+1], verbose=0)[0]
        pred_label = np.argmax(pred_proba)
        confidence = pred_proba[pred_label]
        
        # Determine color
        color = 'green' if pred_label == true_label else 'red'
        
        # Display
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2%}",
            fontsize=9,
            color=color,
            fontweight='bold'
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnt/numbers/digit_predictions.png', dpi=150, bbox_inches='tight')
    print("Prediction visualization saved!\n")
    plt.close()


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    """
    print("Plotting training history...\n")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnt/numbers/training_history.png', dpi=150, bbox_inches='tight')
    print("Training history plot saved!\n")
    plt.close()


def plot_confusion_matrix_heatmap(y_test, y_pred):
    """
    Plot confusion matrix as a heatmap.
    """
    import seaborn as sns
    
    print("Creating confusion matrix heatmap...\n")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix Heatmap', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('mnt/numbers/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix heatmap saved!\n")
    plt.close()


# ============================================================================
# SECTION 7: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline.
    """
    # Step 1: Load fonts
    fonts = load_system_fonts()
    
    # Step 2: Generate dataset
    generator = SyntheticDigitGenerator(fonts, img_size=32)
    X, y = generator.generate_dataset(samples_per_digit=600)
    
    # Step 3: Visualize samples
    visualize_digit_samples(X, y, title="Generated Digit Samples (One Per Class)")
    visualize_random_samples(X, y, n_samples=20, title="Random Generated Samples")
    
    # Step 4: Split dataset
    print("Splitting dataset into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Validation set: {len(X_val)} samples\n")
    
    # Step 5: Build model
    model = build_digit_cnn(input_shape=(32, 32, 1), num_classes=10)
    
    # Step 6: Train model
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=15, batch_size=64
    )
    
    # Step 7: Plot training history
    plot_training_history(history)
    
    # Step 8: Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_val, y_val)
    
    # Step 9: Visualize predictions
    visualize_predictions(model, X_val, y_val, n_samples=10)
    
    # Step 10: Plot confusion matrix heatmap
    try:
        plot_confusion_matrix_heatmap(y_val, y_pred)
    except:
        print("Note: Seaborn not available, skipping heatmap visualization")
    
    # Step 11: Save model
    model_path = 'mnt/numbers/digit_classifier_cnn.keras'
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}\n")
    
    print("=" * 70)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print("\n📁 Generated files:")
    print("  1. digit_samples.png - One sample per digit class")
    print("  2. random_digit_samples.png - Random samples from dataset")
    print("  3. training_history.png - Training curves")
    print("  4. digit_predictions.png - Model predictions visualization")
    print("  5. confusion_matrix.png - Confusion matrix heatmap")
    print("  6. digit_classifier_cnn.h5 - Trained model weights")
    print("\n" + "=" * 70)
    
    # Optional: Integration suggestion
    print("\n💡 OPTIONAL EXTENSION - Multi-Stage OCR Pipeline:")
    print("-" * 70)
    print("""
This digit CNN classifier can be integrated into a multi-stage OCR pipeline:

PIPELINE ARCHITECTURE:

Stage 1: Document Preprocessing
    - Image loading and binarization
    - Noise removal and contrast enhancement
    - Skew correction

Stage 2: Text Region Detection
    - YOLO/Faster R-CNN for text region localization
    - Or traditional methods (connected components, MSER)

Stage 3: Character Segmentation
    - Segment individual characters from text regions
    - Resize segments to 32×32 pixels

Stage 4: Character Classification (This Model)
    - Input: 32×32 grayscale character patch
    - Output: Digit class (0-9) with confidence
    - Use only for numeric fields or filter digit regions

Stage 5: Post-Processing
    - Combine predictions into full text
    - Apply language model for correction
    - Format output (dates, phone numbers, amounts)

EXAMPLE USE CASES:
    - Invoice processing: Extract amounts, dates, invoice numbers
    - Form recognition: Read numeric fields (age, ZIP, phone)
    - License plate recognition: Read plate numbers
    - Check processing: Read amounts and routing numbers
    - Receipt scanning: Extract prices and totals

INTEGRATION TIPS:
    - Use confidence threshold (e.g., >0.9) for reliable predictions
    - Combine with separate alphabet classifier for mixed text
    - Use context (e.g., field type) to improve accuracy
    - Apply ensemble methods for critical applications
    - Fine-tune on domain-specific data when available
    """)
    print("=" * 70)


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()
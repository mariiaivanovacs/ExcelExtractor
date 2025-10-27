#!/usr/bin/env python3
"""
Simple CNN training script for digit recognition (0-9).
This is a simplified version to test the pipeline.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
import json


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


def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    h, w = np_img.shape[:2]
    
    # Handle color images
    if len(np_img.shape) == 3:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    
    up = np_img.copy()
    # up = mask_pixels(up)
    # up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    
    
    # # Step 1: Moderate downsampling (less aggressive than original)
    # if random.random() < 0.6:
    #     scale = random.uniform(0.9, 0.95)  # Less aggressive scaling
    #     small = cv2.resize(up, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    #     up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    
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
    # kernel_size = random.choice([1,3])
    
   
    
    # up = cv2.GaussianBlur(up, (kernel_size, kernel_size), sigmaX=random.uniform(0.5, 0.6))
    
    # up = mask_pixels(up)
    # up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    
    return up

def generate_digit_dataset():
    """Generate a dataset of digit images (0-9) for CNN training."""
    
    # Configuration
    image_size = 32
    font_size = 12
    
    # Simple digits 0-9
    characters = [str(i) for i in range(10)]
    
    print(f"Generating dataset for digits: {characters}")
    
    # Create character to label mapping
    char_to_label = {char: idx for idx, char in enumerate(characters)}
    label_to_char = {idx: char for idx, char in enumerate(characters)}
    
    
    fonts = []
    # Try to use a simple font
    font = ImageFont.truetype("/Users/mariaivanova/Library/Fonts/calibri-regular.ttf")
    fonts.append(font)
    fonts.append(ImageFont.truetype("/Users/mariaivanova/Library/Fonts/Lucida Console Regular.ttf"))
    # # fonts.append(font)
    # font=ImageFont.truetype("/Library/Fonts/adelle-sans-regular.otf")
    # fonts.append(font)
    
    font = ImageFont.truetype("/Users/mariaivanova/Downloads/rainyhearts.ttf")
    fonts.append(font)
    num_variants = 800  # images per digit
    
    # Storage for images and labels
    all_images = []
    all_labels = []
    
    for char_idx, char in enumerate(characters):
        for i in range(num_variants):
            font = random.choice(fonts)

            # Create a blank grayscale image (black background)
            img = Image.new('L', (image_size, image_size), color=255)
            draw = ImageDraw.Draw(img)

            # Compute centered text position
            try:
                text_width, text_height = draw.textbbox((0, 0), char, font=font)[2:4]
            except:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(char, font=font)
            
            position = ((image_size - text_width) // 2, (image_size - text_height) // 2)

            # Draw white text
            draw.text(position, char, fill=80, font=font)

            # Convert to NumPy array
            np_img = np.array(img)

            # Apply random distortion to simulate real-world conditions
            distorted = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)
            
            # Store the image and label
            all_images.append(distorted)
            all_labels.append(char_idx)

        print(f"Processed digit '{char}' ({char_idx + 1}/{len(characters)})")
    
    # Convert to numpy arrays
    images = np.array(all_images, dtype=np.uint8)
    labels = np.array(all_labels, dtype=np.int32)
    
    print(f"Generated dataset: {images.shape} images, {len(np.unique(labels))} classes")
    
    return images, labels, char_to_label, label_to_char

def build_model(input_shape, num_classes):
    """Build a simple CNN model."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


import matplotlib.pyplot as plt
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
    plt.savefig('experiment/digit_samples.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved!\n")
    plt.close()
    
    
def main():
    """Main training function."""
    print("Generating digit dataset...")
    images, labels, char_to_label, label_to_char = generate_digit_dataset()

    visualize_digit_samples(images, labels, title="Generated Digit Samples (One Per Class)")
    
    # Preprocess images for training
    print("Preprocessing images...")
    images = images.astype("float32") / 255.0
    if images.ndim == 3:
        images = images[..., np.newaxis]  # (N, H, W, 1)

    num_classes = len(np.unique(labels))
    input_shape = images.shape[1:]

    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Total samples: {len(images)}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Build and compile model
    model = build_model(input_shape, num_classes)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.15, stratify=labels, random_state=42)

    
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    print("y_train shape:", y_train.shape)
    print("Unique labels:", np.unique(y_train))
    print("Counts:", {int(k): int(v) for k,v in zip(*np.unique(y_train, return_counts=True))})
    # Data augmentation
    data_augment = tf.keras.Sequential([
        layers.RandomRotation(0.03, fill_mode='nearest'),
        layers.RandomTranslation(0.05, 0.05, fill_mode='nearest'),
        layers.RandomContrast(0.1),
        # Optional: slight brightness variation
        layers.Lambda(lambda x: tf.clip_by_value(x + tf.random.uniform([], -0.05, 0.05), 0.0, 1.0))
    ])

    # Build tf.data pipelines
    batch_size = 32
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(1000)
        .map(lambda x, y: (data_augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Train
    print("Starting training...")
    # 1️⃣ Compile FIRST
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",   # since y_train are integers (0–9)
        metrics=["accuracy"]
    )

    # 2️⃣ THEN fit
    history = model.fit(
        train_ds,
        epochs=5,
        validation_data=val_ds,
        verbose=1
    )


    # Save model
    model_path = "models/digit_cnn.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save character mappings
    mappings = {
        'char_to_label': char_to_label,
        'label_to_char': label_to_char,
        'num_classes': num_classes,
        'input_shape': list(input_shape)
    }

    with open("models/digit_mappings.json", "w", encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    print("Character mappings saved to models/digit_mappings.json")
    print("Training completed!")

if __name__ == "__main__":
    main()

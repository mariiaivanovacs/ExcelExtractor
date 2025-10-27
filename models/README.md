# CNN Character Recognition Pipeline

This directory contains a complete pipeline for training CNN models to recognize characters in cell images. The solution bridges the gap between synthetic training data and real-world image quality using sophisticated distortion techniques.

## 🎯 Problem Solved

The main challenge was connecting:
1. **Synthetic character generation** (from `build_characters()` function)
2. **Realistic image distortion** (via `downsample_then_upsample()`)
3. **CNN model training** and inference
4. **Real-world image processing**

## 📁 Files Overview

### Core Training Scripts
- **`train_cnn_tf.py`** - Full character set training (88+ characters including Cyrillic)
- **`train_simple_digits.py`** - Simplified digit-only training (0-9) for testing
- **`usage_example.py`** - Complete pipeline demonstration

### Testing & Evaluation
- **`test_cnn.py`** - Test script for full character model
- **`test_digit_cnn.py`** - Test script for digit-only model

### Generated Files
- **`digit_cnn.h5`** - Trained digit recognition model
- **`digit_mappings.json`** - Character-to-label mappings for digit model
- **`char_cnn_trained.h5`** - Full character model (if trained)
- **`char_mappings.json`** - Mappings for full character model

## 🔧 Key Components

### 1. Distortion Function
```python
def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    """
    Simulates real-world image degradation:
    - Pixelation through downsampling/upsampling
    - Random shifts and jitter
    - Brightness degradation
    - Gaussian blur
    """
```

### 2. Character Generation
```python
def generate_character_image(character, image_size=32, font_size=12, apply_distortion=True):
    """
    Creates synthetic character images:
    - Renders text using system fonts
    - Centers character in image
    - Applies realistic distortions
    """
```

### 3. Model Architecture
```python
def build_model(input_shape, num_classes):
    """
    Simple but effective CNN:
    - 3 Conv2D layers with increasing filters (32→64→128)
    - MaxPooling for downsampling
    - GlobalAveragePooling to reduce parameters
    - Dense layers for classification
    """
```

## 🚀 Quick Start

### 1. Train a Simple Digit Model
```bash
cd /path/to/ExtractExcel
python models/train_simple_digits.py
```

### 2. Test the Model
```bash
python models/test_digit_cnn.py
```

### 3. See Complete Pipeline Demo
```bash
python models/usage_example.py
```

## 📊 Expected Results

The digit model typically achieves:
- **Training accuracy**: ~43% (with heavy distortions)
- **Validation accuracy**: ~53%
- **Real-world performance**: Varies based on image quality

*Note: Lower accuracy is expected due to aggressive distortions that simulate real-world conditions*

## 🔄 Complete Workflow

### For New Character Sets:

1. **Define Characters**
   ```python
   characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
   # or use build_characters() for full set
   ```

2. **Generate Training Data**
   ```python
   images, labels, char_to_label, label_to_char = generate_character_dataset()
   ```

3. **Train Model**
   ```python
   model = build_model(input_shape, num_classes)
   model.fit(train_ds, epochs=10, validation_data=val_ds)
   ```

4. **Use for Prediction**
   ```python
   pred_char, confidence, probs = predict_character_from_image(
       model, image, char_to_label, label_to_char
   )
   ```

### For Real Images:

```python
# Load and preprocess
img = cv2.imread('cell_image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32))

# Predict
model = tf.keras.models.load_model('models/digit_cnn.h5')
with open('models/digit_mappings.json', 'r') as f:
    mappings = json.load(f)

char, conf, probs = predict_character_from_image(
    model, img, mappings['char_to_label'], mappings['label_to_char']
)

print(f"Predicted: '{char}' (confidence: {conf:.3f})")
```

## 🎨 Visualization

The pipeline generates several visualization files:
- **`demo_clean_*.png`** - Clean synthetic characters
- **`demo_distorted_*.png`** - Distorted versions
- **`distortion_comparison.png`** - Side-by-side comparison
- **`test_digit_*.png`** - Test prediction examples

## ⚙️ Configuration Options

### Image Generation
- `image_size`: Output image dimensions (default: 32x32)
- `font_size`: Text rendering size (default: 12)
- `num_variants`: Images per character (default: 50-200)

### Distortion Parameters
- `min_scale/max_scale`: Downsampling range (0.2-0.6)
- `max_shift`: Maximum pixel shift (default: 3)
- `blur_sigma`: Gaussian blur strength (0.5-1.0)

### Model Training
- `batch_size`: Training batch size (default: 32-64)
- `epochs`: Training epochs (default: 5-10)
- `learning_rate`: Optimizer learning rate (default: Adam)

## 🔍 Troubleshooting

### Low Accuracy
- Reduce distortion strength
- Increase training data
- Adjust model architecture
- Check character set complexity

### Memory Issues
- Reduce batch size
- Decrease image resolution
- Use fewer training samples

### Font Issues
- Ensure system fonts are available
- Add fallback to default font
- Test with different font sizes

## 🎯 Integration with Existing Code

This pipeline integrates with:
- **`try_knn_type.py`** - Uses character generation functions
- **`seg_cells.py`** - Processes segmented cell images
- **Feature extraction** - Can be combined with traditional features

The `downsample_then_upsample()` function is the key bridge between synthetic training data and real-world image processing.

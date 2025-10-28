"""
test_digit_model.py

Loads a trained CNN digit classifier (0–9),
runs predictions on all images in a specified directory,
and saves results to experiment/predictions.csv.

Requirements:
    pip install tensorflow pillow pandas
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from pathlib import Path

# ======================
# 🔧 CONFIGURATION
# ======================
MODEL_PATH = "models/fine_tuned_real_data_classifier.keras"     # <-- your trained model path
IMAGES_DIR = "characters"   # <-- change to your test directory
OUTPUT_CSV = "experiment/predictions.csv"
IMG_SIZE = (32, 32)                       # input size used during training
IS_GRAYSCALE = True                       # set False if trained on RGB
# ======================
# 3->9 
# 5_> 6
# 0--> 8
"""
9
2
3
6
0
5
7
1



# """

def preprocess_image(img_path, img_size=(32, 32), grayscale=True):
    """Load and preprocess a single image for CNN prediction."""
    if grayscale:
        img = load_img(img_path, color_mode="grayscale", target_size=img_size)
    else:
        img = load_img(img_path, target_size=img_size)
    
    img_array = img_to_array(img) / 255.0
    if grayscale:
        img_array = np.expand_dims(img_array, axis=-1)  # shape (32,32,1)
    return np.expand_dims(img_array, axis=0)  # batch dimension (1,32,32,1)


def main():
    # Create output directory
    Path(os.path.dirname(OUTPUT_CSV)).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"📦 Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)

    # Collect image paths
    image_paths = sorted([
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    if not image_paths:
        print("⚠️ No images found in the specified directory.")
        return
    
    print("AMount of images: ", len(image_paths))

    results = []
    
    df_numbers = pd.read_csv("data/csv/numbers_new.csv", )

    print(f"🔍 Running predictions on {len(image_paths)} images...")
    for img_path in tqdm(image_paths):
        # get file name 
        filename = os.path.basename(img_path)
        if filename  in df_numbers["filename"].values:
            
            img_tensor = preprocess_image(img_path, IMG_SIZE, IS_GRAYSCALE)
            preds = model.predict(img_tensor, verbose=0)
            predicted_class = np.argmax(preds, axis=1)[0]
            confidence = float(np.max(preds))

            results.append({
                "filename": os.path.basename(img_path),
                "predicted_class": int(predicted_class),
                "confidence": round(confidence, 4)
            })

    # Save predictions
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Predictions saved to: {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()

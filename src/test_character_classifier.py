from tensorflow import keras
import numpy as np
from PIL import Image

def predict_single_patch(model_path, image_path):
    """
    Predict whether a single 32x32 patch contains a number.
    
    Args:
        model_path: Path to trained model (.h5)
        image_path: Path to input image
    
    Returns:
        label: 'NUMBER' or 'OTHER'
        confidence: Probability score (0-1)
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((32, 32))  # Resize to 32x32
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = img_array.reshape(1, 32, 32, 1)  # Add batch & channel dims
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # # Determine label and confidence
    # if prediction > 0.5:
    #     label = 'NUMBER'
    #     confidence = prediction
    # else:
    #     label = 'OTHER'
    #     confidence = 1 - prediction
    
    # return label, confidence
    return prediction

# # Example usage
# if __name__ == '__main__':
#     # label, conf = predict_single_patch('models/digit_cnn.h5', 'characters/cell_r4_c10_blob_1_word_4_char_00.png')
#     prediction = predict_single_patch('models/digit_cnn.h5', 'characters/cell_r4_c10_blob_1_word_4_char_00.png')

#     # print(f"Prediction: {label} (confidence: {conf:.2%})")
#     print("Prediction is: ")
#     print(prediction)
    
    
    
    
from tensorflow import keras
import numpy as np
from PIL import Image
import os

def predict_batch(model_path, image_dir, threshold=0.5):
    """
    Process multiple images in a directory.
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing images
        threshold: Classification threshold (default: 0.5)
    
    Returns
        results: List of dicts with predictions
    """
    # Load model
    model = keras.models.load_model(model_path)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    results = []
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        try:
            # Load and preprocess
            img = Image.open(img_path).convert('L').resize((32, 32))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array.reshape(1, 32, 32, 1)
            
            # Predict
            prob = model.predict(img_array, verbose=0)[0][0]
            
            # Store result
            results.append({
                'filename': img_file,
                'probability': float(prob),
                'label': 'NUMBER' if prob > threshold else 'OTHER',
                'confidence': float(prob if prob > 0.5 else 1 - prob)
            })
        
        except Exception as e:
            results.append({
                'filename': img_file,
                'error': str(e)
            })
    
    return results

# Example usage
if __name__ == '__main__':
    results = predict_batch('mnt/outputs/number_detector_cnn.keras', 'characters')
    
    # create 2 csv files 
    csv_1 = "data/csv/numbers.csv"
    csv_2 = "data/csv/other.csv"
    
    # Print results
    for result in results:
        if 'error' in result:
            print(f"{result['filename']}: ERROR - {result['error']}")
        else:
            print(f"{result['filename']}: {result['label']} "
                  f"(confidence: {result['confidence']:.2%})")
            if result['label'] == 'NUMBER':
                with open(csv_1, 'a') as f:
                    f.write(f"{result['filename']}\n")
            else:
                with open(csv_2, 'a') as f:
                    f.write(f"{result['filename']}\n")
    
    # Filter only NUMBER predictions
    numbers = [r for r in results if r.get('label') == 'NUMBER']
    print(f"\nFound {len(numbers)} number patches out of {len(results)}")
    
    
    # check leght of both csvs
    with open(csv_1, 'r') as f:
        lines = f.readlines()
        print(f"Length of numbers.csv: {len(lines)}")
        
    with open(csv_2, 'r') as f:
        lines = f.readlines()
        print(f"Length of other.csv: {len(lines)}")
        
    
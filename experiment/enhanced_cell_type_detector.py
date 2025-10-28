import cv2, pytesseract, numpy as np
import re
import os

def detect_cell_type_image(cell_img, debug=False):
    """
    Final enhanced cell type detection optimized for your specific data
    """
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    
    if debug:
        print(f"Image stats: mean={gray.mean():.1f}, std={gray.std():.1f}, shape={gray.shape}")
    
    # Check if image is mostly empty (very high mean indicates mostly white)
    if gray.mean() > 250 and gray.std() < 10:
        return "EMPTY"
    
    # Enhanced preprocessing for light text
    processed_images = []
    
    # 1. OTSU thresholding (inverted for dark text on light background)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.append(('OTSU', thresh_otsu))
    
    # 2. Adaptive thresholding
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    processed_images.append(('Adaptive', thresh_adaptive))
    
    # 3. Multiple manual thresholds for very light text
    for threshold in [200, 220, 240]:
        _, thresh_manual = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        processed_images.append((f'Manual_{threshold}', thresh_manual))
    
    # 4. Contrast enhancement + thresholding
    enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=-200)
    _, thresh_enhanced = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
    processed_images.append(('Enhanced', thresh_enhanced))
    
    # 5. Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    for name, img in processed_images[:]:
        cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        processed_images.append((f'{name}_cleaned', cleaned))
    
    # Check for content
    content_found = False
    for name, img in processed_images:
        content_ratio = np.count_nonzero(img) / img.size
        if content_ratio > 0.01:
            content_found = True
            break
    
    if not content_found:
        return "EMPTY"
    
    # Try OCR with number-specific configurations first
    number_configs = [
        '--psm 7 -c tessedit_char_whitelist=0123456789.,',
        '--psm 8 -c tessedit_char_whitelist=0123456789.,',
        '--psm 13 -c tessedit_char_whitelist=0123456789.,',
    ]
    
    text_configs = [
        '--psm 7',
        '--psm 8', 
        '--psm 6',
    ]
    
    # Collect number-specific OCR results
    number_results = []
    for name, img in processed_images:
        for config in number_configs:
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                if text and re.match(r'^[\d\.,\s]+$', text.replace(' ', '')):
                    number_results.append(text)
            except:
                continue
    
    # Collect general OCR results (limited to best preprocessing methods)
    general_results = []
    for name, img in processed_images[:6]:
        for config in text_configs:
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                if text:
                    general_results.append(text)
            except:
                continue
    
    all_texts = number_results + general_results
    
    if not all_texts:
        return "EMPTY"
    
    return classify_text_type_final(all_texts, number_results, debug)

def classify_text_type_final(all_texts, number_results, debug=False):
    """
    Final classification optimized for your data patterns
    """
    if not all_texts:
        return "EMPTY"
    
    # Count consistent number patterns from number-specific OCR
    consistent_numbers = 0
    valid_number_chars = set()
    
    for text in number_results:
        cleaned = re.sub(r'[^\d\.,]', '', text)
        if cleaned and len(cleaned) >= 1:
            digit_count = sum(c.isdigit() for c in cleaned)
            if digit_count >= 1:
                consistent_numbers += 1
                # Track what digits we're seeing
                for c in cleaned:
                    if c.isdigit():
                        valid_number_chars.add(c)
    
    # Count meaningful word patterns (3+ letters, not just OCR noise)
    meaningful_words = 0
    for text in all_texts:
        cleaned = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
        # Look for substantial word-like patterns
        if re.search(r'[a-zA-Zа-яА-Я]{4,}', cleaned):  # 4+ consecutive letters
            meaningful_words += 1
        elif re.search(r'[a-zA-Zа-яА-Я]{3,}', cleaned) and len(cleaned) >= 5:  # 3+ letters in longer text
            meaningful_words += 1

    
    # Optimized decision logic based on your specific data patterns

    # Very strong NUMBER evidence: many consistent number results
    if consistent_numbers >= 5:
        return "NUMBER"

    # Very strong TEXT evidence: multiple substantial words
    if meaningful_words >= 3:
        return "TEXT"

    # Strong NUMBER evidence: consistent numbers with single digit pattern
    if len(valid_number_chars) == 1 and consistent_numbers >= 3:
        return "NUMBER"

    # Strong TEXT evidence: meaningful words with little number evidence
    if meaningful_words >= 2 and consistent_numbers <= 2:
        return "TEXT"

    # Medium NUMBER evidence: some consistent numbers, no meaningful words
    if consistent_numbers >= 3 and meaningful_words == 0:
        return "NUMBER"

    # Medium TEXT evidence: at least one meaningful word
    if meaningful_words >= 1 and consistent_numbers <= 2:
        return "TEXT"

    # Analyze overall character composition for edge cases
    total_digits = 0
    total_letters = 0
    total_chars = 0
    pure_number_texts = 0

    for text in all_texts:
        cleaned = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
        if re.match(r'^\d+[\.,]?\d*$', cleaned) and len(cleaned) >= 1:
            pure_number_texts += 1

        for c in cleaned:
            if c.isdigit():
                total_digits += 1
            elif c.isalpha():
                total_letters += 1
            total_chars += 1

    # if debug:
    #     print(f"Pure number texts: {pure_number_texts}")
    #     if total_chars > 0:
    #         print(f"Overall digit ratio: {total_digits/total_chars:.2f}")

    # Final decision based on composition
    if pure_number_texts >= 2 and meaningful_words == 0:
        return "NUMBER"

    if total_chars > 0:
        digit_ratio = total_digits / total_chars
        if digit_ratio >= 0.6 and meaningful_words == 0:
            return "NUMBER"
        elif digit_ratio <= 0.3 and meaningful_words >= 1:
            return "TEXT"

    # Conservative fallback: slight bias towards TEXT for truly ambiguous cases
    if consistent_numbers >= 2 and meaningful_words == 0:
        return "NUMBER"
    else:
        return "TEXT"


import csv
def test_on_ground_truth():
    """Test the enhanced OCR function against ground truth data
    and save predictions to separate CSV files."""
    
    working_directory = "blobs"
    output_dir = "data/csv"
    os.makedirs(output_dir, exist_ok=True)
    
    numbers_path = os.path.join(output_dir, "numbers_latest.csv")
    others_path = os.path.join(output_dir, "others_latest.csv")
    
    # Initialize counters
    total = 0
    correct = 0
    
    # Open both CSV files for writing
    with open(numbers_path, "w", newline="") as f_num, open(others_path, "w", newline="") as f_oth:
        num_writer = csv.writer(f_num)
        oth_writer = csv.writer(f_oth)
        
        # Write headers
        num_writer.writerow(["filename", "prediction"])
        oth_writer.writerow(["filename", "prediction"])
        
        for filename in os.listdir(working_directory):
            cell_path = os.path.join(working_directory, filename)
            if not filename.lower().endswith('.png'):
                continue
            if not os.path.exists(cell_path):
                continue
            
            img = cv2.imread(cell_path)
            if img is None:
                continue
            
            total += 1
            
            predicted_type = detect_cell_type_image(img, debug=False)
            # print(f"{filename:25}  Predicted: {predicted_type}")
            
            # Write to the appropriate CSV file
            if predicted_type.upper() == "NUMBER":
                num_writer.writerow([filename, predicted_type])
            else:
                oth_writer.writerow([filename, predicted_type])
    
    print(f"\n✅ Done. Saved predictions to:\n  → {numbers_path}\n  → {others_path}")
    print(f"Processed {total} images.\n")
    
    return correct / total if total > 0 else 0

if __name__ == "__main__":
    test_on_ground_truth()



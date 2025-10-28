import cv2, pytesseract, numpy as np
import re
import os

def detect_cell_type_image(cell_img, debug=False):
    """
    Enhanced cell type detection with better handling of light characters
    and improved number/text classification
    """
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    if debug:
        print(f"Image stats: mean={gray.mean():.1f}, std={gray.std():.1f}, shape={gray.shape}")

    # Check if image is mostly empty (very high mean indicates mostly white)
    if gray.mean() > 250 and gray.std() < 10:
        return "EMPTY"

    # Enhanced preprocessing for light text
    processed_images = []

    # 1. OTSU thresholding
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed_images.append(('OTSU', thresh_otsu))

    # 2. Adaptive thresholding (inverted for dark text on light background)
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
    processed_images.append(('Adaptive', thresh_adaptive))

    # 3. Multiple manual thresholds for very light text
    for threshold in [200, 220, 240]:
        _, thresh_manual = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        processed_images.append((f'Manual_{threshold}', thresh_manual))

    # 4. Contrast enhancement + thresholding
    enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=-200)  # Increase contrast
    _, thresh_enhanced = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY_INV)
    processed_images.append(('Enhanced', thresh_enhanced))

    # 5. Morphological operations to clean up
    kernel = np.ones((2,2), np.uint8)
    for name, img in processed_images[:]:
        # Remove noise
        cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # Fill gaps
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        processed_images.append((f'{name}_cleaned', cleaned))

    # Check for content in processed images
    content_found = False
    for name, img in processed_images:
        content_ratio = np.count_nonzero(img) / img.size
        if debug:
            print(f"{name}: content_ratio={content_ratio:.4f}")
        if content_ratio > 0.01:  # At least 1% content
            content_found = True

    if not content_found:
        return "EMPTY"

    # Try OCR with different configurations
    texts = []

    # OCR configurations optimized for numbers vs text
    number_configs = [
        '--psm 7 -c tessedit_char_whitelist=0123456789.,',
        '--psm 8 -c tessedit_char_whitelist=0123456789.,',
        '--psm 13 -c tessedit_char_whitelist=0123456789.,',
    ]

    text_configs = [
        '--psm 7',
        '--psm 8',
        '--psm 6',
        '--psm 13',
    ]

    # Try number-specific OCR first
    number_results = []
    for name, img in processed_images:
        for config in number_configs:
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                if text and re.match(r'^[\d\.,\s]+$', text.replace(' ', '')):
                    number_results.append(text)
                    if debug:
                        print(f"Number OCR ({name}): '{text}'")
            except:
                continue

    # Try general OCR
    general_results = []
    for name, img in processed_images[:6]:  # Use only best preprocessing methods
        for config in text_configs:
            try:
                text = pytesseract.image_to_string(img, config=config).strip()
                if text:
                    general_results.append(text)
                    if debug:
                        print(f"General OCR ({name}): '{text}'")
            except:
                continue

    # Combine results
    all_texts = number_results + general_results

    if debug:
        print(f"All OCR results: {all_texts}")

    if not all_texts:
        return "EMPTY"

    # Enhanced classification
    return classify_text_type_enhanced(all_texts, number_results, debug)

def classify_text_type_enhanced(all_texts, number_results, debug=False):
    """
    Enhanced classification using both general OCR and number-specific OCR results
    """
    if not all_texts:
        return "EMPTY"

    # More conservative approach for number classification
    strong_number_evidence = 0
    strong_text_evidence = 0

    # Check number-specific results more carefully
    if number_results:
        valid_numbers = []
        for text in number_results:
            cleaned = re.sub(r'[^\d\.,]', '', text)
            # Only count as valid number if it has substantial numeric content
            if cleaned and len(cleaned) >= 1:
                digit_count = sum(c.isdigit() for c in cleaned)
                if digit_count >= 1 and len(cleaned) <= 10:  # Reasonable number length
                    valid_numbers.append(cleaned)

        if valid_numbers:
            # Check if we have multiple consistent number results
            if len(valid_numbers) >= 2:
                strong_number_evidence += 3
            else:
                strong_number_evidence += 1

            if debug:
                print(f"Valid number results: {valid_numbers}")

    # Analyze general OCR results
    word_like_count = 0
    number_like_count = 0

    for text in all_texts:
        cleaned = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')

        if not cleaned or len(cleaned) < 1:
            continue

        if debug:
            print(f"Analyzing: '{cleaned}'")

        # Check for clear word patterns (multiple letters in sequence)
        if re.search(r'[a-zA-Zа-яА-Я]{3,}', cleaned):
            word_like_count += 1
            strong_text_evidence += 2
            if debug:
                print(f"  -> Strong TEXT evidence (word pattern)")

        # Check for clear number patterns
        if re.match(r'^\d+[\.,]?\d*$', cleaned) and len(cleaned) >= 1:
            number_like_count += 1
            strong_number_evidence += 2
            if debug:
                print(f"  -> Strong NUMBER evidence (pure number)")

        # Calculate ratios
        digit_count = sum(c.isdigit() for c in cleaned)
        letter_count = sum(c.isalpha() for c in cleaned)
        total_chars = len(cleaned)

        digit_ratio = digit_count / total_chars if total_chars > 0 else 0
        letter_ratio = letter_count / total_chars if total_chars > 0 else 0

        if debug:
            print(f"  digit_ratio={digit_ratio:.2f}, letter_ratio={letter_ratio:.2f}")

        # Conservative scoring
        if digit_ratio >= 0.8 and digit_count >= 1:
            strong_number_evidence += 1
        elif letter_ratio >= 0.6 and letter_count >= 2:
            strong_text_evidence += 1

    if debug:
        print(f"Evidence - NUMBER: {strong_number_evidence}, TEXT: {strong_text_evidence}")
        print(f"Counts - word_like: {word_like_count}, number_like: {number_like_count}")

    # Use the best performing logic from our testing
    consistent_numbers = len([x for x in number_results if re.match(r'^[\d\.,]+$', re.sub(r'[^\d\.,]', '', x)) and len(re.sub(r'[^\d\.,]', '', x)) >= 1])

    if debug:
        print(f"Consistent number results: {consistent_numbers}")

    # Optimized decision logic based on testing results
    if consistent_numbers >= 4:
        return "NUMBER"
    elif word_like_count >= 3:
        return "TEXT"
    elif consistent_numbers >= 3 and word_like_count <= 1:
        return "NUMBER"
    elif word_like_count >= 2 and consistent_numbers <= 1:
        return "TEXT"
    elif strong_number_evidence >= 8 and strong_text_evidence <= 12:
        return "NUMBER"
    elif strong_text_evidence >= 18:
        return "TEXT"
    elif number_like_count >= 3 and word_like_count == 0:
        return "NUMBER"
    else:
        return "TEXT"

def classify_text_type(texts, debug=False):
    """
    Classify extracted texts as NUMBER or TEXT with improved logic
    """
    number_votes = 0
    text_votes = 0

    for text in texts:
        # Clean the text
        cleaned = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')

        if not cleaned:
            continue

        if debug:
            print(f"Analyzing: '{cleaned}'")

        # Enhanced number detection patterns
        number_patterns = [
            r'^\d+$',                    # Pure digits: 123
            r'^\d+\.\d+$',              # Decimal: 123.45
            r'^\d+,\d+$',               # Comma decimal: 123,45
            r'^\d+[\.,]\d+$',           # Either comma or dot: 123.45 or 123,45
            r'^\d+[\s\.,]*\d*$',        # Numbers with spaces/separators: 1 234.56
            r'^[\d\s\.,]+$',            # Only digits, spaces, dots, commas
        ]

        # Check if it matches number patterns
        is_number = False
        for pattern in number_patterns:
            if re.match(pattern, cleaned):
                is_number = True
                break

        # Additional checks for numbers
        if not is_number:
            # Remove common OCR artifacts and check again
            cleaned_artifacts = re.sub(r'[^\d\.,]', '', cleaned)
            if cleaned_artifacts and re.match(r'^[\d\.,]+$', cleaned_artifacts):
                # Check if it's mostly digits
                digit_ratio = sum(c.isdigit() for c in cleaned_artifacts) / len(cleaned_artifacts)
                if digit_ratio > 0.7:  # At least 70% digits
                    is_number = True

        if is_number:
            number_votes += 1
            if debug:
                print(f"  -> NUMBER vote")
        else:
            text_votes += 1
            if debug:
                print(f"  -> TEXT vote")

    if debug:
        print(f"Final votes - NUMBER: {number_votes}, TEXT: {text_votes}")

    # Decision logic
    if number_votes == 0 and text_votes == 0:
        return "EMPTY"
    elif number_votes > text_votes:
        return "NUMBER"
    elif text_votes > number_votes:
        return "TEXT"
    else:
        # Tie - use additional heuristics
        # If any text contains letters, it's probably TEXT
        for text in texts:
            if re.search(r'[a-zA-Zа-яА-Я]', text):
                return "TEXT"
        # Otherwise default to NUMBER
        return "NUMBER"
   
def test_on_ground_truth():
    """
    Test the enhanced OCR function against ground truth data
    """
    import os

    # Load ground truth data
    ground_truth = {}
    with open("data/input/dictionary.txt", "r") as f:
        for line in f:
            if "," in line:
                filename, cell_type = line.strip().split(", ")
                # Normalize the cell type
                cell_type = cell_type.upper()
                if cell_type == "NUMBER":
                    cell_type = "NUMBER"
                ground_truth[filename] = cell_type

    print(f"Loaded {len(ground_truth)} ground truth samples")
    print("="*60)

    correct = 0
    total = 0
    errors = []

    for filename, expected_type in ground_truth.items():
        cell_path = os.path.join("cells_cleaned", filename)
        if not os.path.exists(cell_path):
            print(f"Warning: {filename} not found")
            continue

        img = cv2.imread(cell_path)
        if img is None:
            print(f"Warning: Could not load {filename}")
            continue

        predicted_type = detect_cell_type_image(img, debug=False)
        total += 1

        if predicted_type == expected_type:
            correct += 1
            status = "✅ CORRECT"
        else:
            errors.append((filename, expected_type, predicted_type))
            status = "❌ WRONG"

        print(f"{filename:20} | Expected: {expected_type:6} | Predicted: {predicted_type:6} | {status}")

    print("="*60)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for filename, expected, predicted in errors:
            print(f"  {filename}: {expected} -> {predicted}")

    return correct/total if total > 0 else 0

def test_single_cell(filename, debug=True):
    """
    Test a single cell with debug output
    """
    # cell_path = os.path.join("cells_cleaned", filename)
    # img = cv2.imread(cell_path)
    # if img is None:
    #     print(f"Could not load {filename}")
    #     return None
    
    img = cv2.imread("cells_cleaned/cell_r11_c13.png")
    cell_type = detect_cell_type_image(img, debug=False)
    print(f"Cell type: {cell_type}")  # Returns: "NUMBER", "TEXT", or "EMPTY"

    print(f"Testing {filename}:")
    result = detect_cell_type_image(img, debug=debug)
    print(f"Result: {result}")
    return result

if __name__ == "__main__":
    # Test on ground truth data
    # test_on_ground_truth()

    print("\n" + "="*60)
    print("DETAILED TEST ON PROBLEMATIC CASES:")
    print("="*60)
    img = cv2.imread("cells_cleaned/cell_r11_c13.png")
    detect_cell_type_image(img, debug=True)
    
    

    # # Test some specific cases with debug
    # test_cases = ["cell_r0_c1.png", "cell_r1_c1.png", "cell_r3_c11.png"]
    # for case in test_cases:
    #     print(f"\n--- {case} ---")
    #     test_single_cell(case, debug=True)
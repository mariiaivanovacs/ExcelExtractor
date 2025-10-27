import os
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytesseract

# If on Windows and Tesseract isn't on PATH, uncomment and set the path:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

IN_FOLDER = Path("words_production")
MAX_FILES = 10000
TOKEN_CSV = "results/ocr_tokens_first10.csv"
SUMMARY_CSV = "results/ocr_summary_first10.csv"

# Regexes for classification
# Allow Cyrillic (А-Я, а-я, Ё, ё) and Latin letters
RE_NUMBER = re.compile(r'^[+-]?\d+([.,]\d+)*$')
RE_WORD = re.compile(r'^[A-Za-zА-Яа-яЁё.,!?;:()\-\']+$')


def preprocess_for_ocr(img):
    """
    img: BGR image from cv2.imread
    Returns a grayscale preprocessed image ready for OCR.
    Steps:
      - convert to gray
      - upscale (makes OCR easier for small fonts)
      - histogram equalization (contrast)
      - bilateral filter to reduce noise while preserving edges
      - adaptive threshold (binarize)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale: 200x32 -> scale factor 3 or 4 (experiment). Using 3 to keep speed.
    scale = 3
    new_w = int(gray.shape[1] * scale)
    new_h = int(gray.shape[0] * scale)
    gray_up = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Contrast equalization
    gray_eq = cv2.equalizeHist(gray_up)

    # Denoise while preserving edges
    denoised = cv2.bilateralFilter(gray_eq, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive threshold (works better than simple Otsu on uneven illumination)
    th = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    # Optionally perform a small morphological open to remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th


def ocr_tokens_from_image(img_preprocessed):
    """
    Run pytesseract on preprocessed image and return list of token dicts:
    [{'text':..., 'conf':..., 'left':..., 'top':..., 'width':..., 'height':...}, ...]
    Uses pytesseract.image_to_data for per-token info.
    """
    custom_config = r"-l rus --oem 1 --psm 6"  # adjust language or psm as needed
    data = pytesseract.image_to_data(img_preprocessed, output_type=pytesseract.Output.DICT, config=custom_config)

    tokens = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text == "":
            continue  # skip empty
        conf_str = str(data['conf'][i])
        conf = float(conf_str) if re.match(r'^-?\d+(\.\d+)?$', conf_str) else None        
        token = {
            "text": text,
            "conf": conf,
            "left": int(data['left'][i]),
            "top": int(data['top'][i]),
            "width": int(data['width'][i]),
            "height": int(data['height'][i])
        }
        tokens.append(token)
    return tokens


def classify_token(token_text):
    t = token_text.strip()
    if RE_NUMBER.match(t):
        return "number"
    if RE_WORD.match(t):
        return "word"
    # Mixed tokens like 'A1', 'e-3', punctuation, or non-latin letters
    return "other"


def main():
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted([p for p in IN_FOLDER.iterdir() if p.suffix.lower() in exts and p.is_file()])
    files = files[:MAX_FILES]

    rows_tokens = []
    rows_summary = []

    for fp in files:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"Warning: failed to read {fp}")
            continue

        pre = preprocess_for_ocr(img)
        tokens = ocr_tokens_from_image(pre)

        count_word = 0
        count_number = 0
        count_other = 0

        for tk in tokens:
            typ = classify_token(tk["text"])
            if typ == "word":
                count_word += 1
            elif typ == "number":
                count_number += 1
            else:
                count_other += 1

            rows_tokens.append({
                "filename": fp.name,
                "token": tk["text"],
                "token_type": typ,
                "confidence": tk["conf"],
                "left": tk["left"],
                "top": tk["top"],
                "width": tk["width"],
                "height": tk["height"]
            })

        # decide dominant label
        if (count_word + count_number + count_other) == 0:
            dominant = "none"
        elif count_word > count_number and count_word >= count_other:
            dominant = "words"
        elif count_number > count_word and count_number >= count_other:
            dominant = "numbers"
        elif count_word == 0 and count_number == 0 and count_other > 0:
            dominant = "other"
        else:
            dominant = "mixed"

        rows_summary.append({
            "filename": fp.name,
            "num_tokens": len(tokens),
            "num_words": count_word,
            "num_numbers": count_number,
            "num_other": count_other,
            "dominant": dominant
        })

        print(f"Processed {fp.name}: tokens={len(tokens)} (words={count_word}, numbers={count_number}, other={count_other})")

    # Save results
    if rows_tokens:
        df_tokens = pd.DataFrame(rows_tokens)
        df_tokens.to_csv(TOKEN_CSV, index=False)
        print(f"Saved token-level results to {TOKEN_CSV}")
    else:
        print("No tokens detected in the selected images.")

    df_summary = pd.DataFrame(rows_summary)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved image-level summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

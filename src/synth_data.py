#!/usr/bin/env python3
"""
synth_ocr_generator_dual.py
Generate synthetic Russian words and characters images with augmentations.
Creates two separate datasets:
  1. Words recognition dataset (2-3 samples per word)
  2. Characters recognition dataset (3 samples per character combination)
Output: 
  - synthetic_data/words/images/ and synthetic_data/words/labels.csv
  - synthetic_data/characters/images/ and synthetic_data/characters/labels.csv
"""

import os
import csv
import argparse
import random
import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_HEIGHT = 32
DEFAULT_MAX_WIDTH = 160
RANDOM_SEED = 42

DEFAULT_WIDTHS = [12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 132, 140, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252, 260, 268, 276, 284, 292, 300, 308, 316, 324, 332, 340, 348]
DEFAULT_HEIGHTS = [12, 15, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
DEFAULT_FONTS = [
    "/System/Library/Fonts/Times New Roman.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Verdana.ttf",
    "/System/Library/Fonts/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
]


# -----------------------------
# Helper functions
# -----------------------------
def random_choice_font(font_paths: List[str], base_size=32):
    """Pick a random font from available paths"""
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, base_size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_text_as_image(text: str, font_path_list: List[str], base_height=DEFAULT_HEIGHT, pad=6):
    """Render text to a grayscale PIL Image"""
    font = random_choice_font(font_path_list, base_size=int(base_height * 0.8))
    
    # Create dummy image to measure text size
    has_lower = any(c.islower() for c in text)

    dummy = Image.new("L", (10, 10), color=255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Create actual image with padding
    w = max(w + pad * 2, 4)
    h = max(h + pad * 2, int(base_height * 0.8))
    if has_lower:
        pad_top = int(base_height * 0.1)
        pad_bottom = int(base_height * 0.1)
        h += pad_top + pad_bottom
        img = Image.new("L", (w, h), color=255)  # white background
        draw = ImageDraw.Draw(img)
        draw.text((pad, pad // 2), text, fill=0, font=font)

    else:
        img = Image.new("L", (w, h), color=255)  # white background
        draw = ImageDraw.Draw(img)
        draw.text((pad, pad // 2), text, fill=0, font=font)
    
    return img


def rescale_to_target(img_pil: Image.Image, target_height=DEFAULT_HEIGHT, max_width=DEFAULT_MAX_WIDTH):
    """Scale image to target height maintaining aspect ratio, then pad/crop width"""
    w, h = img_pil.size
    new_h = target_height
    new_w = int(w * (new_h / h))
    img = img_pil.resize((new_w, new_h), resample=Image.BICUBIC)
    
    if new_w > max_width:
        img = img.resize((max_width, target_height), resample=Image.BICUBIC)
        return img
    
    # Pad to max_width
    pad_left = random.randint(0, max(0, max_width - new_w))
    pad_right = max_width - new_w - pad_left
    img = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=255)
    return img


# -----------------------------
# Augmentation functions (NO ROTATION)
# -----------------------------
def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    """Simulate low-res by shrinking then upscaling"""
    h, w = np_img.shape
    scale = random.uniform(0.2, 0.3)
    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))
    small = cv2.resize(np_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return up


def gaussian_blur(np_img: np.ndarray, max_sigma=1.5):
    """Apply Gaussian blur"""
    k = random.uniform(0.0, max_sigma)
    if k <= 0.01:
        return np_img
    ksize = int(max(1, round(k * 3)) * 2 + 1)
    out = cv2.GaussianBlur(np_img, (ksize, ksize), sigmaX=k)
    return out


def motion_blur(np_img: np.ndarray, max_ksize=7, p=0.15):
    """Apply motion blur"""
    if random.random() > p:
        return np_img
    k = random.randint(3, max_ksize)
    kernel = np.zeros((k, k))
    if random.choice([True, False]):
        kernel[int((k - 1) / 2), :] = np.ones(k)  # horizontal
    else:
        kernel[:, int((k - 1) / 2)] = np.ones(k)  # vertical
    kernel = kernel / k
    out = cv2.filter2D(np_img, -1, kernel)
    return out


def salt_and_pepper(np_img: np.ndarray, amount=0.005):
    """Add salt and pepper noise"""
    out = np_img.copy()
    h, w = out.shape
    num_salt = np.ceil(amount * h * w * random.uniform(0.5, 1.5))
    coords = (np.random.randint(0, h, int(num_salt)), np.random.randint(0, w, int(num_salt)))
    out[coords] = 0 if random.random() < 0.5 else 255
    return out


def brightness_contrast(np_img: np.ndarray, brightness_delta=0.3, contrast_range=(0.7, 1.3)):
    """Adjust brightness and contrast"""
    img = np_img.astype(np.float32) / 255.0
    b = random.uniform(-brightness_delta, brightness_delta)
    img = img + b
    c = random.uniform(contrast_range[0], contrast_range[1])
    img = (img - 0.5) * c + 0.5
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def random_rect_occlusion(np_img: np.ndarray, max_rect_h=8, p=0.2):
    """Add random rectangular occlusion"""
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    rect_w = random.randint(1, min(w // 3, 20))
    rect_h = random.randint(1, min(max_rect_h, h // 2))
    x = random.randint(0, max(0, w - rect_w))
    y = random.randint(0, max(0, h - rect_h))
    c = random.choice([0, 255, 128])
    out = np_img.copy()
    out[y:y + rect_h, x:x + rect_w] = c
    return out


def elastic_distortion(np_img: np.ndarray, alpha=30, sigma=4, p=0.25):
    """Apply elastic distortion"""
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    dx = (np.random.rand(h, w) * 2 - 1).astype(np.float32)
    dy = (np.random.rand(h, w) * 2 - 1).astype(np.float32)
    ksize = int(max(1, sigma))
    dx = cv2.GaussianBlur(dx, (ksize | 1, ksize | 1), sigma)
    dy = cv2.GaussianBlur(dy, (ksize | 1, ksize | 1), sigma)
    dx = dx * alpha
    dy = dy * alpha
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)
    out = cv2.remap(np_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def binarization_variation(np_img: np.ndarray, p=0.3):
    """Apply binarization (thresholding)"""
    if random.random() > p:
        return np_img
    if random.random() < 0.5:
        out = cv2.adaptiveThreshold(np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, out = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def add_background_texture(np_img: np.ndarray, strength=0.12, p=0.4):
    """Add paper-like texture"""
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    noise = np.random.randn(h, w).astype(np.float32)
    k = int(max(1, min(h, w) * 0.02))
    noise = cv2.GaussianBlur(noise, (k | 1, k | 1), sigmaX=k / 2.0)
    noise = cv2.normalize(noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    alpha = random.uniform(0.02, strength)
    out = cv2.addWeighted(np_img.astype(np.float32), 1.0, noise.astype(np.float32), alpha, 0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def gaussian_gradient_illumination(np_img: np.ndarray, p=0.25):
    """Apply gradient illumination"""
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(xs, ys)
    sigma = random.uniform(0.4, 1.0)
    mask = np.exp(- (xv ** 2 + yv ** 2) / (2 * (sigma ** 2)))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    gamma = random.uniform(0.6, 1.3)
    out = np_img.astype(np.float32) * (mask ** gamma)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def resize_np_image(np_img, size=(10, 10)):
    return np.array(Image.fromarray(np_img).resize(size, Image.Resampling.LANCZOS))
# -----------------------------
# Main generation logic
# -----------------------------
def generate_one_image(text: str, fonts: List[str], out_size=(DEFAULT_MAX_WIDTH, DEFAULT_HEIGHT)):
    """
    Render text and apply augmentation pipeline (NO ROTATION)
    Returns final PIL Image
    """
    has_lower = any(c.islower() for c in text)
    if has_lower:
        base_height = 35
    else:
        base_height = 40
        
    out_size = (DEFAULT_WIDTHS[len(text)-1], DEFAULT_HEIGHTS[len(text)-1])
    
    # Render text
    pil_img = render_text_as_image(text, fonts,base_height = base_height )
    # pil_img = rescale_to_target(pil_img, target_height=out_size[1], max_width=out_size[0])
    
    # Convert to numpy
    np_img = np.array(pil_img).astype(np.uint8)
    
    # Apply augmentations (NO ROTATION)
    # 1) Downsample/upsample for low-res effect
    
    np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)
    np_img = resize_np_image(np_img, size=(int(out_size[0]*0.85), int(out_size[1]*0.85)))

    # 2) Blur
    # np_img = gaussian_blur(np_img, max_sigma=1.8)
    # np_img = motion_blur(np_img, max_ksize=7, p=0.15)
    
    # 3) Brightness/contrast
    # np_img = brightness_contrast(np_img, brightness_delta=0.1, contrast_range=(0.6, 1))
    
    # 4) Noise
    # np_img = salt_and_pepper(np_img, amount=random.uniform(0.001, 0.01))
    
    # 5) Occlusions
    # np_img = random_rect_occlusion(np_img, max_rect_h=10, p=0.15)
    
    # # 6) Elastic distortion (NO AFFINE/ROTATION)
    # np_img = elastic_distortion(np_img, alpha=random.uniform(20, 40), sigma=random.uniform(3, 6), p=0.25)
    
    # # 7) Background texture / gradient
    # np_img = add_background_texture(np_img, strength=0.12, p=0.4)
    # np_img = gaussian_gradient_illumination(np_img, p=0.25)
    
    # # 8) Binarization
    # np_img = binarization_variation(np_img, p=0.3)
    
    # 9) Final clamp
    # np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


# -----------------------------
# CSV reading
# -----------------------------
def read_csv_data(path: str) -> List[str]:
    """Read data from CSV file, one item per line"""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # If CSV with multiple columns, take first column
            if ',' in s and len(s.split(',')) > 1:
                s = s.split(',')[0].strip()
            items.append(s)
    return items


def find_usable_fonts(preferred: List[str]) -> List[str]:
    """Find available fonts from preferred list"""
    found = []
    for p in preferred:
        if os.path.exists(p):
            found.append(p)
    return found


# -----------------------------
# Dataset generation functions
# -----------------------------
def generate_words_dataset(words: List[str], fonts: List[str], out_dir: str, 
                          samples_per_word=3, max_width=160, height=32):
    """Generate synthetic dataset for words recognition"""
    imgs_dir = os.path.join(out_dir, "images")
    os.makedirs(imgs_dir, exist_ok=True)
    
    labels_path = os.path.join(out_dir, "labels.csv")
    
    print(f"\n[Words Dataset] Generating {len(words)} words with {samples_per_word} samples each...")
    
    with open(labels_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'label'])
        
        for word in tqdm(words, desc="Generating words"):
            # Generate multiple samples per word
            for _ in range(samples_per_word):
                img = generate_one_image(word, fonts, out_size=(max_width, height))
                 
                fname = f"{uuid.uuid4().hex}.png"
                outpath = os.path.join(imgs_dir, fname)
                img.save(outpath)
                csvwriter.writerow([os.path.join("images", fname), word])
    
    print(f"[Words Dataset] Done! Total images: {len(words) * samples_per_word}")
    print(f"  Images: {imgs_dir}")
    print(f"  Labels: {labels_path}")


def generate_characters_dataset(characters: List[str], fonts: List[str], out_dir: str,
                               samples_per_char=3, max_width=70, height=32):
    """Generate synthetic dataset for character recognition"""
    imgs_dir = os.path.join(out_dir, "images")
    os.makedirs(imgs_dir, exist_ok=True)
    
    labels_path = os.path.join(out_dir, "labels.csv")
    
    print(f"\n[Characters Dataset] Generating {len(characters)} character combinations with {samples_per_char} samples each...")
    
    with open(labels_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'label'])
        
        for char_combo in tqdm(characters, desc="Generating characters"):
            # Generate multiple samples per character combination
            for _ in range(samples_per_char):
                img = generate_one_image(char_combo, fonts, out_size=(max_width, height))

                fname = f"{uuid.uuid4().hex}.png"
                outpath = os.path.join(imgs_dir, fname)
                img.save(outpath)
                csvwriter.writerow([os.path.join("images", fname), char_combo])
    
    print(f"[Characters Dataset] Done! Total images: {len(characters) * samples_per_char}")
    print(f"  Images: {imgs_dir}")
    print(f"  Labels: {labels_path}")


# -----------------------------
# Main
# -----------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Find available fonts
    fonts = find_usable_fonts(args.fonts)
    if len(fonts) == 0:
        print("[WARN] No preferred fonts found; will use PIL default fonts fallback.")
    else:
        print(f"[INFO] Using fonts: {fonts}")
    
    # Create base output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Generate words dataset
    if args.words_csv:
        words = []
        with open(args.words_csv, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:  # ignore blank lines
                    words.append(word)
        if len(words) == 0:
            print(f"[WARN] No words found in {args.words_csv}")
        else:
            words_out_dir = os.path.join(args.out_dir, "words")
            generate_words_dataset(
                words, 
                fonts, 
                words_out_dir,
                samples_per_word=args.samples_per_word,
                max_width=args.max_width,
                height=args.height
            )
    
    # Generate characters dataset
    if args.characters_csv:
        characters = read_csv_data(args.characters_csv)
        if len(characters) == 0:
            print(f"[WARN] No characters found in {args.characters_csv}")
        else:
            chars_out_dir = os.path.join(args.out_dir, "characters")
            generate_characters_dataset(
                characters,
                fonts,
                chars_out_dir,
                samples_per_char=args.samples_per_char,
                # max_width=args.max_width,
                # height=args.height
            )
    
    print("\n✓ All datasets generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic OCR datasets for words and characters recognition"
    )
    parser.add_argument(
        "--words_csv",
        type=str,
        help="CSV file with Russian words (one per line)"
    )
    parser.add_argument(
        "--characters_csv",
        type=str,
        help="CSV file with character combinations (one per line)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="synthetic_data",
        help="Output directory (will create 'words' and 'characters' subdirs)"
    )
    parser.add_argument(
        "--samples_per_word",
        type=int,
        default=3,
        help="Number of samples to generate per word (default: 3)"
    )
    parser.add_argument(
        "--samples_per_char",
        type=int,
        default=3,
        help="Number of samples to generate per character combination (default: 3)"
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=160,
        help="Canvas width in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Canvas height in pixels"
    )
    parser.add_argument(
        "--fonts",
        nargs="*",
        default=DEFAULT_FONTS,
        help="List of font file paths (TTF)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one CSV is provided
    if not args.words_csv and not args.characters_csv:
        parser.error("At least one of --words_csv or --characters_csv must be provided")
    
    main(args)
    
    
    # python src/synth_data.py  --characters_csv data/input/char.csv  --out_dir synthetic_data  --samples_per_word 1
    
    # both
    # python src/synth_data.py --words_csv data/input/russian_words.csv --characters_csv data/input/characters.csv  --out_dir synthetic_data  --samples_per_word 3  --samples_per_char 3
    
    
    
    #python src/synth_data.py  --words_csv data/input/russian_pos_tokens.csv  --out_dir synthetic_data  --samples_per_word 2

#!/usr/bin/env python3
"""
synth_ocr_generator.py
Generate synthetic Russian words/numbers images with many augmentations.
Output: images in out_dir/images/ and labels CSV out_dir/labels.csv
"""

import os
import csv
import argparse
import random
import math
import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import cv2
from tqdm import tqdm

# -----------------------------
# Helper / config
# -----------------------------
DEFAULT_HEIGHT = 32           # canonical height (we will downsample to small sizes later)
DEFAULT_MAX_WIDTH = 160
TARGET_WORD_SIZE = (60, 20)   # visually similar target used in synthesis step (for guidance)
RANDOM_SEED = 42

# Default font paths - edit if these paths don't exist on your system.
DEFAULT_FONTS = [
    "/System/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Times New Roman.ttf",
    "/System/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Verdana.ttf",
    "/System/Library/Fonts/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Lucida Grande.ttf",
    "/System/Library/Fonts/Supplemental/Palatino.ttc",
]

# Characters to include when generating numeric templates and tokens
NUMERIC_TEMPLATES = [
    "{d}", "{d}{d}", "{d}{d}{d}", "{d}{d}{d}{d}",
    "{d}.{d}{d}", "{d}.{d}", ".{d}{d}", "{d}{d}00", "{d}00"
]


# -----------------------------
# Augmentation functions
# -----------------------------
def random_choice_font(font_paths: List[str], base_size=32):
    # pick a font path that exists, else use PIL default font
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, base_size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_text_as_image(text: str, font_path_list: List[str], base_height=DEFAULT_HEIGHT, pad=6, mode='separate_chars', jitter_kerning=1):
    """
    Render text to a grayscale PIL Image.
    mode:
      - 'normal' : render whole text with PIL draw.text
      - 'separate_chars' : render characters individually (allows controlled overlap to simulate merged glyphs)
    """
    font = random_choice_font(font_path_list, base_size=int(base_height*0.8))
    # estimate size by measuring text
    # check if in the text is lowercase
    has_lower = any(c.islower() for c in text)
    if has_lower:
        base_height = int(base_height * 1.2)
    
    dummy = Image.new("L", (10, 10), color=255)
    draw = ImageDraw.Draw(dummy)
    if mode == 'normal':
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # w, h = draw.textsize(text, font=font)
    else:
        # sum widths for separate rendering
        w = 0
        max_h = 0
        for ch in text:
            bbox = draw.textbbox((0, 0), ch, font=font)
            ch_w = bbox[2] - bbox[0]
            ch_h = bbox[3] - bbox[1]
            # ch_w, ch_h = draw.textsize(ch, font=font)
            w += ch_w + random.randint(-jitter_kerning, jitter_kerning)
            max_h = max(max_h, ch_h)
        h = max_h
    w = max(w + pad*2, 4)
    h = max(h + pad*2, int(base_height*0.8))

    # if has_lower means cut height from the top and add to botton
    
        
    if mode == 'normal':
        draw.text((pad, pad//2), text, fill=0, font=font)
    else:
        # render each char with optional small overlaps to simulate merging
        x = pad
        for i, ch in enumerate(text):
            ch_font = random_choice_font(font_path_list, base_size=int(base_height*0.8))
            # ch_w, ch_h = draw.textsize(ch, font=ch_font)
            bbox = draw.textbbox((0, 0), ch, font=ch_font)
            ch_w = bbox[2] - bbox[0]
            ch_h = bbox[3] - bbox[1]
            overlap = random.randint(-int(ch_w * 0.35), int(ch_w * 0.05)) if random.random() < 0.25 else 0
            draw.text((x + overlap, pad//2 + random.randint(-1, 1)), ch, fill=0, font=ch_font)
            x += ch_w + random.randint(-jitter_kerning, jitter_kerning)
    return img


def rescale_to_target(img_pil: Image.Image, target_height=DEFAULT_HEIGHT, max_width=DEFAULT_MAX_WIDTH):
    # maintain aspect ratio, scale height then possibly pad/truncate width
    w, h = img_pil.size
    new_h = target_height
    new_w = int(w * (new_h / h))
    img = img_pil.resize((new_w, new_h), resample=Image.BICUBIC)
    if new_w > max_width:
        # optionally downscale width a bit to fit
        img = img.resize((max_width, target_height), resample=Image.BICUBIC)
        return img
    # pad to max_width a bit randomly to increase width variety
    pad_left = random.randint(0, max(0, max_width - new_w))
    pad_right = max_width - new_w - pad_left
    img = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=255)
    return img


def downsample_then_upsample(np_img: np.ndarray, min_scale=0.2, max_scale=0.6):
    # simulate low-res: shrink then upscale (bicubic)
    h, w = np_img.shape
    scale = random.uniform(min_scale, max_scale)
    small_h = max(1, int(h * scale))
    small_w = max(1, int(w * scale))
    small = cv2.resize(np_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return up


def gaussian_blur(np_img: np.ndarray, max_sigma=1.5):
    k = random.uniform(0.0, max_sigma)
    if k <= 0.01:
        return np_img
    # OpenCV uses kernel size, approximate from sigma
    ksize = int(max(1, round(k*3))*2 + 1)
    out = cv2.GaussianBlur(np_img, (ksize, ksize), sigmaX=k)
    return out


def motion_blur(np_img: np.ndarray, max_ksize=9, p=0.15):
    if random.random() > p:
        return np_img
    k = random.randint(3, max_ksize)
    # create motion blur kernel
    kernel = np.zeros((k, k))
    if random.choice([True, False]):
        kernel[int((k-1)/2), :] = np.ones(k)  # horizontal
    else:
        kernel[:, int((k-1)/2)] = np.ones(k)  # vertical
    kernel = kernel / k
    out = cv2.filter2D(np_img, -1, kernel)
    return out


def salt_and_pepper(np_img: np.ndarray, amount=0.005):
    out = np_img.copy()
    h, w = out.shape
    num_salt = np.ceil(amount * h * w * random.uniform(0.5, 1.5))
    coords = (np.random.randint(0, h, int(num_salt)), np.random.randint(0, w, int(num_salt)))
    out[coords] = 0 if random.random() < 0.5 else 255
    return out


def brightness_contrast(np_img: np.ndarray, brightness_delta=0.3, contrast_range=(0.7, 1.3)):
    img = np_img.astype(np.float32) / 255.0
    # brightness
    b = random.uniform(-brightness_delta, brightness_delta)
    img = img + b
    # contrast
    c = random.uniform(contrast_range[0], contrast_range[1])
    img = (img - 0.5) * c + 0.5
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def random_rect_occlusion(np_img: np.ndarray, max_rect_h=8, p=0.2):
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    rect_w = random.randint(1, min(w//3, 20))
    rect_h = random.randint(1, min(max_rect_h, h//2))
    x = random.randint(0, max(0, w - rect_w))
    y = random.randint(0, max(0, h - rect_h))
    c = random.choice([0, 255, 128])
    out = np_img.copy()
    out[y:y+rect_h, x:x+rect_w] = c
    return out


def elastic_distortion(np_img: np.ndarray, alpha=34, sigma=4, p=0.25):
    if random.random() > p:
        return np_img
    # OpenCV-based elastic transform (fast enough)
    h, w = np_img.shape
    dx = (np.random.rand(h, w) * 2 - 1).astype(np.float32)
    dy = (np.random.rand(h, w) * 2 - 1).astype(np.float32)
    ksize = int(max(1, sigma))
    dx = cv2.GaussianBlur(dx, (ksize|1, ksize|1), sigma)
    dy = cv2.GaussianBlur(dy, (ksize|1, ksize|1), sigma)
    dx = dx * alpha
    dy = dy * alpha
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)
    out = cv2.remap(np_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


def affine_transform(np_img: np.ndarray, max_rot=3.0, max_shear=0.02, p=0.5):
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    # rotation
    rot = random.uniform(-max_rot, max_rot)
    # small shear by moving top corners
    pts1 = np.float32([[0,0],[w,0],[0,h]])
    dx = random.uniform(-max_shear*w, max_shear*w)
    dy = random.uniform(-max_shear*h, max_shear*h)
    pts2 = np.float32([[dx,dy],[w+dx, -dy],[dx, h+dy]])
    M = cv2.getAffineTransform(pts1, pts2)
    out = cv2.warpAffine(np_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # apply rotation as well
    M2 = cv2.getRotationMatrix2D((w//2, h//2), rot, 1.0)
    out = cv2.warpAffine(out, M2, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out


def binarization_variation(np_img: np.ndarray, p=0.4):
    if random.random() > p:
        return np_img
    # adaptive thresholding or Otsu
    if random.random() < 0.5:
        out = cv2.adaptiveThreshold(np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, out = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def add_background_texture(np_img: np.ndarray, strength=0.12, p=0.5):
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    # paper-like texture: Perlin-like via gaussian filtered noise
    noise = np.random.randn(h, w).astype(np.float32)
    k = int(max(1, min(h, w) * 0.02))
    noise = cv2.GaussianBlur(noise, (k|1, k|1), sigmaX=k/2.0)
    noise = cv2.normalize(noise, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    alpha = random.uniform(0.02, strength)
    out = cv2.addWeighted(np_img.astype(np.float32), 1.0, noise.astype(np.float32), alpha, 0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def gaussian_gradient_illumination(np_img: np.ndarray, p=0.3):
    if random.random() > p:
        return np_img
    h, w = np_img.shape
    # create radial gaussian mask
    xs = np.linspace(-1, 1, w)
    ys = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(xs, ys)
    sigma = random.uniform(0.4, 1.0)
    mask = np.exp(- (xv**2 + yv**2) / (2 * (sigma**2)))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    gamma = random.uniform(0.6, 1.3)
    out = np_img.astype(np.float32) * (mask**gamma)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# -----------------------------
# Merge neighbor characters simulation (advanced)
# -----------------------------
def render_with_merge(text: str, font_paths: List[str], base_height=DEFAULT_HEIGHT):
    """
    Render by placing characters and intentionally overlapping some neighbors to create merged characters.
    This is more sophisticated than simple whole-text rendering.
    """
    # render each char on its own canvas then paste with overlap
    char_imgs = []
    for ch in text:
        img = render_text_as_image(ch, font_paths, base_height=base_height, mode='normal')
        char_imgs.append(img)
    # compute total width with random negative spacing
    heights = [im.size[1] for im in char_imgs]
    total_h = max(heights)
    x = 0
    # compute approximate widths
    placements = []
    for i, im in enumerate(char_imgs):
        w, h = im.size
        # sometimes overlap chars by 20-40% to simulate merge
        overlap = int(w * random.uniform(0.0, 0.35)) if random.random() < 0.4 else 0
        placements.append((x - overlap, im))
        x += (w - overlap) + random.randint(-1, 2)
    # compose onto background
    out_w = max(1, x + 6)
    out_h = total_h + 6
    canvas = Image.new("L", (out_w, out_h), color=255)
    for px, im in placements:
        y = random.randint(0, max(0, out_h - im.size[1]))
        canvas.paste(im, (max(0, px), y), None)
    return canvas


# -----------------------------
# Main generation logic
# -----------------------------
def create_numeric_sample(max_digits=6):
    # pick a template and fill with digits
    t = random.choice(NUMERIC_TEMPLATES)
    d = lambda: str(random.randint(0, 9))
    # replace {d} tokens with digits
    s = ""
    for ch in t:
        if ch == "{":
            s += ""  # handled below
    # simpler approach:
    out = ""
    i = 0
    while i < len(t):
        if t[i:i+3] == "{d}":
            out += d()
            i += 3
        else:
            out += t[i]
            i += 1
    # occasionally produce multi-digit sequences
    if random.random() < 0.25:
        out = "".join(str(random.randint(0, 9)) for _ in range(random.randint(1, max_digits)))
    return out


def generate_one_image(text: str, fonts: List[str], out_size=(DEFAULT_MAX_WIDTH, DEFAULT_HEIGHT)):
    """
    Render text, then apply a pipeline of augmentations and return final PIL Image.
    """
    # Choose mode (normal, separate_chars, merge)
    mode = random.choices(['normal', 'separate_chars', 'merge'], weights=[0.4, 0.4, 0.2])[0]
    if mode == 'merge':
        pil_img = render_with_merge(text, fonts, base_height=out_size[1])
    else:
        pil_img = render_text_as_image(text, fonts, base_height=out_size[1], mode=('separate_chars' if mode == 'separate_chars' else 'normal'))
    # scale to canonical height then add augmentations
    pil_img = rescale_to_target(pil_img, target_height=out_size[1], max_width=out_size[0])
    # convert to numpy gray
    np_img = np.array(pil_img).astype(np.uint8)
    # composite augmentation pipeline (random order for variability)
    # 1) downsample/upscale to simulate tiny chars
    if random.random() < 0.7:
        np_img = downsample_then_upsample(np_img, min_scale=0.15, max_scale=0.6)
    # 2) blur(s)
    np_img = gaussian_blur(np_img, max_sigma=1.8)
    np_img = motion_blur(np_img, max_ksize=9, p=0.12)
    # 3) brightness/contrast
    np_img = brightness_contrast(np_img, brightness_delta=0.25, contrast_range=(0.6, 1.4))
    # 4) noise
    np_img = salt_and_pepper(np_img, amount=random.uniform(0.001, 0.01))
    # 5) occlusions
    np_img = random_rect_occlusion(np_img, max_rect_h=10, p=0.18)
    # 6) elastic/affine
    np_img = elastic_distortion(np_img, alpha=random.uniform(20, 40), sigma=random.uniform(3, 6), p=0.28)
    np_img = affine_transform(np_img, max_rot=3.0, max_shear=0.04, p=0.5)
    # 7) background texture / gradient
    np_img = add_background_texture(np_img, strength=0.12, p=0.45)
    np_img = gaussian_gradient_illumination(np_img, p=0.28)
    # 8) final binarization sometimes
    np_img = binarization_variation(np_img, p=0.32)
    # 9) final clamp and return as PIL
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


# -----------------------------
# CLI & orchestration
# -----------------------------
def read_words_csv(path: str) -> List[str]:
    # accept simple newline-separated or CSV, strip empty lines
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # if CSV with many columns, take first column
            if ',' in s and len(s.split(',')) > 1:
                s = s.split(',')[0].strip()
            words.append(s)
    return words


def find_usable_fonts(preferred: List[str]) -> List[str]:
    found = []
    for p in preferred:
        if os.path.exists(p):
            found.append(p)
    # if none found, let PIL try its default by returning empty list (random_choice_font will fallback)
    return found


def main(args):
    #  each character - 33 characters 
    # punctuation - 10 characters 
    #  all combinations of 2 characters - 1089
    # all combination of 3 characters - 25937
    #  all combinations of 2 numeric values - 100 
    # all comb of dot and numbers 100 + 100 + 100 + 10 
    
    #     "{d}", "{d}{d}", "{d}{d}{d}", "{d}{d}{d}{d}",
    # "{d}.{d}{d}", "{d}.{d}", ".{d}{d}", "{d}{d}00", "{d}00"
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    imgs_dir = os.path.join(args.out_dir, "images")
    os.makedirs(imgs_dir, exist_ok=True)

    words = read_words_csv(args.words_csv)
    print("\n\n\n")
    print(len(words))
    
    if len(words) == 0:
        raise RuntimeError("No words found in words.csv")

    fonts = find_usable_fonts(args.fonts)
    if len(fonts) == 0:
        print("[WARN] No preferred fonts found; will use PIL default fonts fallback. To improve realism, pass font paths with --fonts")
    else:
        print(f"[INFO] Using fonts: {fonts}")

    # create labels CSV
    labels_path = os.path.join(args.out_dir, "labels.csv")
    csvfile = open(labels_path, 'w', newline='', encoding='utf-8')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['filename', 'label'])

    # sample distribution: words vs numeric templates
    num_images = args.num_images
    p_numeric = args.numeric_ratio  # fraction of numeric samples
    p_word = 1.0 - p_numeric

    for i in tqdm(range(num_images), desc="Generating"):
        # choose source
        if random.random() < p_numeric:
            text = create_numeric_sample(max_digits=args.max_numeric_digits)
            # sometimes prefix/suffix with words from spec to simulate mixed tokens
            if random.random() < 0.1 and random.random() < 0.5 and len(words) > 0:
                w = random.choice(words)
                if random.random() < 0.5:
                    text = w + " " + text
                else:
                    text = text + " " + w
        else:
            # sample word with random upper/lower casing
            w = random.choice(words)
            if random.random() < 0.5:
                # randomly change case of first letter or full
                if random.random() < 0.6:
                    # capitalize first letter
                    w = w.capitalize()
                else:
                    # random per-char case
                    w = ''.join(ch.upper() if random.random() < 0.5 else ch.lower() for ch in w)
            text = w

        # generate image
        img = generate_one_image(text, fonts, out_size=(args.max_width, args.height))

        # final optional resize to mimic your target 60x20 sometimes (downsample then ups)
        if random.random() < args.force_small_ratio:
            # downscale to a very small size and upscale back to height to produce extremely low-res glyphs
            small_w = max(10, int(args.target_small_w * random.uniform(0.8, 1.2)))
            small_h = max(6, int(args.target_small_h * random.uniform(0.8, 1.2)))
            img_small = img.resize((small_w, small_h), Image.BILINEAR)
            img = img_small.resize((args.max_width, args.height), Image.BICUBIC)

        # final crop/pad randomness (small vertical jitter)
        # Save
        fname = f"{uuid.uuid4().hex}.png"
        outpath = os.path.join(imgs_dir, fname)
        img.save(outpath)
        csvwriter.writerow([os.path.join("images", fname), text])

    csvfile.close()
    print("Done. Images in:", imgs_dir)
    print("Labels CSV:", labels_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Russian OCR dataset generator with augmentations")
    parser.add_argument("--words_csv", type=str, required=True, help="CSV or newline file with Russian words (one per line)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (images + labels.csv)")
    parser.add_argument("--num_images", type=int, default=10000, help="Total number of synthetic images to generate")
    parser.add_argument("--max_width", type=int, default=160, help="Canvas width (px)")
    parser.add_argument("--height", type=int, default=32, help="Canvas height (px)")
    parser.add_argument("--fonts", nargs="*", default=DEFAULT_FONTS, help="List of font file paths to try (TTF). Will fallback to PIL default font if none found.")
    parser.add_argument("--numeric_ratio", type=float, default=0.25, help="Fraction of numeric samples vs words")
    parser.add_argument("--max_numeric_digits", type=int, default=6, help="Max digits when creating numeric sequences")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--force_small_ratio", type=float, default=0.35, help="Fraction of samples to force extremely small/resampled artifacts (simulate 3-4px chars)")
    parser.add_argument("--target_small_w", type=int, default=60, help="target small width used before upscaling in forced-small samples")
    parser.add_argument("--target_small_h", type=int, default=20, help="target small height used before upscaling in forced-small samples")
    args = parser.parse_args()
    main(args)

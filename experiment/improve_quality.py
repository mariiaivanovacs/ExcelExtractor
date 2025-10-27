import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

# -------------------------
# Helper image transforms
# -------------------------

def upsample(img: np.ndarray, target_size=(200,32), interp=cv2.INTER_CUBIC) -> np.ndarray:
    """Upsample grayscale or color image to target_size using chosen interpolation."""
    return cv2.resize(img, target_size, interpolation=interp)

def ensure_gray(img: np.ndarray) -> np.ndarray:
    """Return single-channel uint8 grayscale image."""
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        # scale floats [0..1] to 0..255
        img = (255 * (img.astype(np.float32) - img.min()) / max(1e-6, img.max() - img.min())).astype(np.uint8)
    return img

def clahe_equalize(img: np.ndarray, clipLimit=2.0, tileGridSize=(8,8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

def denoise_fastnl(img: np.ndarray, h=10) -> np.ndarray:
    """Non-local means denoising (good for small-speckle noise)."""
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)

def bilateral(img: np.ndarray, d=9, sigmaColor=75, sigmaSpace=75) -> np.ndarray:
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

def gaussian_blur(img: np.ndarray, ksize=(3,3), sigma=0) -> np.ndarray:
    return cv2.GaussianBlur(img, ksize, sigma)

def median_blur(img: np.ndarray, k=3) -> np.ndarray:
    return cv2.medianBlur(img, k)

def unsharp_mask(img: np.ndarray, kernel_size=(3,3), sigma=1.0, amount=1.0, threshold=0):
    """Sharpen with unsharp mask. img must be uint8 grayscale."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.abs(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def sobel_edges(img: np.ndarray) -> np.ndarray:
    """Compute combined gradient magnitude (0..255 uint8)."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = (255 * (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)).astype(np.uint8)
    return mag

def contrast_stretch(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, (2, 98))
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # optional linear stretch between p2/p98
    out = np.clip((out - p2) * 255.0 / max(1, (p98 - p2)), 0, 255).astype(np.uint8)
    return out

# -------------------------
# Main preprocess pipeline
# -------------------------

def preprocess(
    img: np.ndarray,
    target_size: Tuple[int, int] = (200,32),
    do_upsample: bool = True,
    use_lanczos: bool = False,
    apply_clahe: bool = True,
    denoise_method: str = "fastnl",  # choices: 'fastnl', 'bilateral', None
    do_unsharp: bool = True,
    add_edge_channel: bool = True,
    add_hf_channel: bool = True,
    save_debug: Optional[str] = "experiment"
) -> np.ndarray:
    """
    Preprocess image for models when input height is very small (~9-10 px).
    Returns a 3-channel uint8 image (stacked features) suitable for CNNs
    or standard single-channel uint8 if you set add_edge_channel/add_hf_channel False.
    """

    # ensure grayscale uint8
    gray = ensure_gray(img)

    # 1) Upsample to a larger size (bicubic or Lanczos)
    interp = cv2.INTER_LANCZOS4 if use_lanczos else cv2.INTER_CUBIC
    if do_upsample:
        up = upsample(gray, target_size=target_size, interp=interp)
    else:
        up = cv2.resize(gray, target_size, interpolation=interp)

    # 2) Contrast enhancement
    # if apply_clahe:
    #     enhanced = clahe_equalize(up, clipLimit=2.0, tileGridSize=(8, 8))
    # else:
    #     enhanced = contrast_stretch(up)

    # # 3) Denoise (choose best for your noise type)
    # if denoise_method == "fastnl":
    #     den = denoise_fastnl(enhanced, h=8)
    # elif denoise_method == "bilateral":
    #     den = bilateral(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    # else:
    #     den = enhanced
    
    # den = enhanced

    # # 4) Mild smoothing to remove tiny artifacts BUT keep edges
    # den = gaussian_blur(den, ksize=(3, 3), sigma=0)

    # # 5) Sharpen to enhance high-frequency details (unsharp)
    # if do_unsharp:
    #     sharp = unsharp_mask(den, kernel_size=(3,3), sigma=1.0, amount=0.8, threshold=1)
    # else:
    #     sharp = den

    # # 6) Create derived channels
    # channels = []
    # # channel 0: the main preprocessed image
    # channels.append(sharp)

    # # channel 1: edge map (Sobel)
    # if add_edge_channel:
    #     edges = sobel_edges(sharp)
    #     channels.append(edges)

    # # channel 2: high-frequency content (original - blurred)
    # if add_hf_channel:
    #     blurred_for_hf = cv2.GaussianBlur(sharp, (9, 9), 0)
    #     hf = cv2.subtract(sharp, blurred_for_hf)
    #     # normalize hf to 0..255
    #     hf = (255 * (hf.astype(np.float32) - hf.min()) / max(1e-9, hf.max() - hf.min())).astype(np.uint8)
    #     channels.append(hf)

    # # If we didn't create 3 channels, pad with duplicates so we always return 3 channels
    # while len(channels) < 3:
    #     channels.append(channels[-1].copy())

    # stacked = cv2.merge(channels[:3])

    # # optional debug save
    # save_debug = "experiment"
   
    # # os.makedirs(save_debug, exist_ok=True)
    # cv2.imwrite(os.path.join(save_debug, "01_up.png"), up)
    # cv2.imwrite(os.path.join(save_debug, "02_enhanced.png"), enhanced)
    # cv2.imwrite(os.path.join(save_debug, "03_denoised.png"), den)
    # cv2.imwrite(os.path.join(save_debug, "04_sharp.png"), sharp)
    # cv2.imwrite(os.path.join(save_debug, "05_edges.png"), channels[1] if add_edge_channel else channels[0])
    # cv2.imwrite(os.path.join(save_debug, "06_hf.png"), channels[2] if add_hf_channel else channels[0])
    # cv2.imwrite(os.path.join(save_debug, "07_stack.png"), stacked)

    return up

# -------------------------
# Augmentation helper (optional)
# -------------------------

def generate_variants(img: np.ndarray, n_variants: int = 6, target_size=(200,32)):
    """Return list of multiple processed variants for training (augment and preprocess)."""
    variants = []
    # base preprocess
    variants.append(preprocess(img, target_size=target_size, use_lanczos=False, denoise_method="fastnl", save_debug=None))
    # try different denoising / sharpening settings
    variants.append(preprocess(img, target_size=target_size, use_lanczos=True, denoise_method="bilateral", do_unsharp=False))
    variants.append(preprocess(img, target_size=target_size, use_lanczos=True, denoise_method=None, do_unsharp=True))
    # add small geometric augmentations on original then preprocess
    for angle in [-6, 6]:
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        variants.append(preprocess(rot, target_size=target_size, use_lanczos=False))
    # contrast/gamma variants
    gamma = 0.8
    adjusted = np.clip((img.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    variants.append(preprocess(adjusted, target_size=target_size))
    return variants

# -------------------------
# Example usage
# -------------------------
import os
import cv2

if __name__ == "__main__":
    working_directory = "words_production"
    os.makedirs("experiment/debug_out", exist_ok=True)

    count = 0
    files_explored = []

    for filename in os.listdir(working_directory):
        if not filename.lower().endswith('.png'):
            continue

        # limit number of files (optional)
        # if count > 10:
        #     break

        cell_path = os.path.join(working_directory, filename)
        base_name = os.path.splitext(filename)[0]  # correct way

        img = cv2.imread(cell_path, cv2.IMREAD_GRAYSCALE)
        processed = preprocess(img, target_size=(200, 32), save_debug="experiment/debug_out")

        out_path = os.path.join("words_production", f"{base_name}.png")
        cv2.imwrite(out_path, processed)

        files_explored.append(filename)
        count += 1


    # # create several augmented variants
    # variants = generate_variants(img, target_size=(200,32))
    # for i, v in enumerate(variants):
    #     cv2.imwrite(f"experiment/variant_{i}.png", v)

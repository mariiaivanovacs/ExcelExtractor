import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional



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


    return up

# -------------------------
# Augmentation helper (optional)
# -------------------------

# def generate_variants(img: np.ndarray, n_variants: int = 6, target_size=(200,32)):
#     """Return list of multiple processed variants for training (augment and preprocess)."""
#     variants = []
#     # base preprocess
#     variants.append(preprocess(img, target_size=target_size, use_lanczos=False, denoise_method="fastnl", save_debug=None))
#     # try different denoising / sharpening settings
#     variants.append(preprocess(img, target_size=target_size, use_lanczos=True, denoise_method="bilateral", do_unsharp=False))
#     variants.append(preprocess(img, target_size=target_size, use_lanczos=True, denoise_method=None, do_unsharp=True))
#     # add small geometric augmentations on original then preprocess
#     for angle in [-6, 6]:
#         M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
#         rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
#         variants.append(preprocess(rot, target_size=target_size, use_lanczos=False))
#     # contrast/gamma variants
#     gamma = 0.8
#     adjusted = np.clip((img.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
#     variants.append(preprocess(adjusted, target_size=target_size))
#     return variants

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


        cell_path = os.path.join(working_directory, filename)
        base_name = os.path.splitext(filename)[0]  # correct way

        img = cv2.imread(cell_path, cv2.IMREAD_GRAYSCALE)
        processed = preprocess(img, target_size=(200, 32), save_debug="experiment/debug_out")

        out_path = os.path.join("words_production", f"{base_name}.png")
        cv2.imwrite(out_path, processed)

        files_explored.append(filename)
        count += 1


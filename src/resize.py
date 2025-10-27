#!/usr/bin/env python3
"""
Pad images horizontally (left & right) with white so their width becomes TARGET_W.
Also pad images vertically (top & bottom) with white so their height becomes TARGET_H
(if the image is shorter). If either dimension is already >= target, that dimension is
left unchanged. Saves results either in-place (overwrite=True) or into a subfolder `words_resized`.
"""

from pathlib import Path
from PIL import Image, ImageOps

TARGET_W = 100  # desired width (pixels). Width will be padded if smaller.
TARGET_H = 16   # desired height (pixels). Height will be padded symmetrically if smaller.

def process_image_pad_lr(in_path: Path, out_path: Path, target_w: int = TARGET_W, target_h: int = TARGET_H, bg_color=(255,255,255)):
    """
    Load image, flatten alpha to white if present, then pad left/right/top/bottom symmetrically
    so final width >= target_w and final height >= target_h. If a dimension is already >= its
    target, that dimension is not changed. If both dimensions are already >= targets, image is copied.
    """
    img = Image.open(in_path)

    # Handle alpha channel (flatten to white)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, bg_color + (255,))
        bg.paste(img, mask=img.split()[3])
        img = bg.convert("RGB")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    orig_w, orig_h = img.size

    # compute horizontal padding (symmetric) if needed
    if orig_w < target_w:
        total_pad_w = target_w - orig_w
        pad_left = total_pad_w // 2
        pad_right = total_pad_w - pad_left
    else:
        pad_left = pad_right = 0

    # compute vertical padding (symmetric) if needed
    if orig_h < target_h:
        total_pad_h = target_h - orig_h
        pad_top = total_pad_h // 2
        pad_bottom = total_pad_h - pad_top
    else:
        pad_top = pad_bottom = 0

    # if no padding required in either dimension, just copy
    if pad_left == pad_right == pad_top == pad_bottom == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return {"in": in_path.name, "out": out_path.name, "orig_size": (orig_w, orig_h), "action": "copied"}

    # choose correct fill color depending on mode
    if img.mode == "L":
        fill_color = 255  # single-channel white
    else:
        fill_color = bg_color  # RGB white

    # apply padding: border=(left, top, right, bottom)
    padded = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill_color)

    new_w, new_h = padded.size
    # sanity checks: new dims should be >= original dims and meet targets when originals were smaller
    assert new_w >= orig_w and new_h >= orig_h
    if orig_w < target_w:
        assert new_w == target_w
    if orig_h < target_h:
        assert new_h == target_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    padded.save(out_path)

    return {
        "in": in_path.name,
        "out": out_path.name,
        "orig_size": (orig_w, orig_h),
        "padding": (pad_left, pad_top, pad_right, pad_bottom),
        "new_size": (new_w, new_h),
        "action": "padded"
    }


def main(folder: Path, pattern: str = "*.*", target_w: int = TARGET_W, target_h: int = TARGET_H):
    folder = folder.resolve()
    out_folder = Path("words_production")  # convert string to Path

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    files = [p for p in folder.glob(pattern) if p.suffix.lower() in exts and p.is_file()]

    if not files:
        print("No image files found in", folder)
        return

    print(f"Found {len(files)} images. Writing results to: {out_folder}")

    out_folder.mkdir(parents=True, exist_ok=True)

    for p in files:
        out_name = p.name
        out_path = out_folder / out_name  # always save in words_production

        try:
            info = process_image_pad_lr(p, out_path, target_w=target_w, target_h=target_h)
            if info["action"] == "padded":
                print(f"Padded: {info['in']:30s} -> {info['out']:30s}  orig={info['orig_size']} pad={info['padding']} new={info['new_size']}")
            else:
                print(f"Copied: {info['in']:30s} -> {info['out']:30s}  orig={info['orig_size']}")
        except Exception as e:
            print(f"⚠️  Failed on {p.name}: {e}")


# if __name__ == "__main__":
#     main(Path("words"))



if __name__ == "__main__":
    main(Path("words"))

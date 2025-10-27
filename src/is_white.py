import os
import cv2
import numpy as np

def calculate_column_intensity(img: np.ndarray,
                               ignore_vertical_border_frac: float = 0.0) -> np.ndarray:
    """
    Compute normalized per-column darkness values in [0,1].
    Dark pixels => values close to 1. White/background => values close to 0.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image (uint8) or color (BGR) image.
    ignore_vertical_border_frac : float
        Fraction of rows to ignore from top and bottom (0.0..0.45).
        Useful to ignore cell borders/lines near edges.

    Returns
    -------
    column_intensities : np.ndarray, shape (width,)
        Float values in [0,1] describing average dark amount per column.
    """
    # convert to grayscale if needed
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # convert to float in [0,1], where 0=black,1=white
    img_float = gray.astype(np.float32) / 255.0

    h, w = img_float.shape
    if h == 0 or w == 0:
        return np.zeros((w,), dtype=float)

    # optionally crop vertical borders (to avoid header/footer lines)
    if ignore_vertical_border_frac > 0:
        top = int(round(h * ignore_vertical_border_frac))
        bottom = h - top
        # guard against degenerate cropping
        if bottom <= top:
            cropped = img_float
        else:
            cropped = img_float[top:bottom, :]
    else:
        cropped = img_float

    # inverted so dark => high
    inverted = 1.0 - cropped  # in [0,1]

    # average darkness per column
    column_intensities = np.mean(inverted, axis=0)  # shape (w,)

    # numerical safety: clip to [0,1]
    column_intensities = np.clip(column_intensities, 0.0, 1.0)

    return column_intensities


def remove_white_cells_from_folder(path: str,
                                   col_darkness_thresh: float = 0.1,
                                   min_white_col_frac: float = 0.9,
                                   ignore_vertical_border_frac: float = 0.03,
                                   verbose: bool = True):
    """
    Iterate PNG files in `path`, check column intensities and remove files
    that are mostly white (>= min_white_col_frac of columns have darkness < col_darkness_thresh).

    Parameters
    ----------
    path : str
        Folder with .png images.
    col_darkness_thresh : float
        Column darkness threshold in [0,1]. Columns with value < this are considered "white".
    min_white_col_frac : float
        If fraction of white columns >= this, file is considered white and will be removed.
    ignore_vertical_border_frac : float
        Fraction of rows to ignore at top/bottom when computing column intensities.
    verbose : bool
        Print progress info.
    """
    if not os.path.isdir(path):
        raise ValueError(f"Path not found: {path}")

    all_files = [f for f in os.listdir(path) if f.lower().endswith('.png')]
    removed = []

    if verbose:
        print(f"Found {len(all_files)} .png files in {path}")

    for filename in all_files:
        img_path = os.path.join(path, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                if verbose:
                    print(f"Warning: cannot read {filename}, skipping")
                continue

            # make grayscale (handles RGBA too)
            if img.ndim == 3 and img.shape[2] == 4:
                # convert RGBA -> RGB first (drop alpha)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if img.ndim == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

            cols = calculate_column_intensity(img_gray,
                                              ignore_vertical_border_frac=ignore_vertical_border_frac)

            # decide white columns
            white_cols_mask = cols < col_darkness_thresh
            white_frac = float(np.sum(white_cols_mask)) / max(1, cols.size)

            if verbose:
                print(f"{filename}: white_frac={white_frac:.3f}, cols={cols.size}, "
                      f"mean_dark={cols.mean():.3f}, max_dark={cols.max():.3f}")

            if white_frac > min_white_col_frac:
                try:
                    os.remove(img_path)
                    removed.append(filename)
                    if verbose:
                        print(f" -> Removed {filename} (white_frac {white_frac:.3f} >= {min_white_col_frac})")
                except Exception as e:
                    if verbose:
                        print(f"Error removing {filename}: {e}")

        except Exception as e:
            if verbose:
                print(f"Error processing {filename}: {e}")

    if verbose:
        print(f"Removed {len(removed)} files.")
    return removed


if __name__ == "__main__":
    # Example usage: adjust path and thresholds as needed
    folder_path = "words_production"
    removed_files = remove_white_cells_from_folder(
        path=folder_path,
        col_darkness_thresh=0.1,      # column considered empty if darkness < 0.1
        min_white_col_frac=0.9,       # remove file when >=80% columns are empty
        ignore_vertical_border_frac=0.03,  # ignore 3% top and bottom rows
        verbose=True
    )

    print("Files removed:", removed_files)
    print("Remaining files:", len([f for f in os.listdir(folder_path) if f.lower().endswith('.png')]))

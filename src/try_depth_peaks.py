import numpy as np
import cv2
from typing import Dict, Any, Tuple


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Convert image to single-channel float32 in [0,1]."""
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr = img.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def column_profile_inverted(img: np.ndarray) -> np.ndarray:
    """
    Column profile where dark ink -> larger values.
    Returns 1D array length = width.
    """
    gray = _to_gray_float(img)
    col_mean = gray.mean(axis=0)       # white ~1.0
    prof = 1.0 - col_mean              # ink high
    return prof.astype(np.float32)


def count_peaks_and_deep_valleys(
    img: np.ndarray,
    smooth_win: int = 3,
    min_peak_prominence: float = 0.02,
    depth_threshold_frac: float = 0.35,
    abs_depth_min: float = 0.02,
    return_profile: bool = False,
) -> Dict[str, Any]:
    """
    Compute peaks and deep-valley metrics on the column profile.

    Args:
      img: input image (H,W) or (H,W,3) uint8 or float.
      smooth_win: integer odd window for moving average smoothing (>=1). Use 1 to disable.
      min_peak_prominence: minimal prominence (in normalized [0..1] profile) to count a peak.
                           Very small peaks caused by noise will be ignored.
      depth_threshold_frac: fraction of global (max-min) peak range used to decide "deep" valley.
      abs_depth_min: additional absolute min depth (in 0..1) to consider deep (useful if range small).
      return_profile: if True, return 'profile' (numpy array) in dict.

    Returns dict:
      {
        "peaks_count": int,
        "valleys_count": int,
        "deep_valleys_count": int,
        "deep_valley_ratio": float,  # deep_valleys_count / max(1, peaks_count)
        "avg_valley_depth": float,   # average of (min(left_peak,right_peak)-valley), 0..1
        "avg_deep_depth": float,     # average depth only for deep valleys (0 if none)
        "profile_range": float,      # max-min of profile
        "profile": np.ndarray (optional)
      }
    """
    prof = column_profile_inverted(img)  # ink -> high

    # smoothing
    if smooth_win is None or int(smooth_win) <= 1:
        prof_s = prof
    else:
        k = int(smooth_win)
        if k % 2 == 0:
            k += 1
        # simple moving average
        kernel = np.ones(k, dtype=np.float32) / k
        prof_s = np.convolve(prof, kernel, mode='same')

    n = prof_s.size
    if n < 3 or np.allclose(prof_s, prof_s[0]):
        return {
            "peaks_count": 0,
            "valleys_count": 0,
            "deep_valleys_count": 0,
            "deep_valley_ratio": 0.0,
            "avg_valley_depth": 0.0,
            "avg_deep_depth": 0.0,
            "profile_range": 0.0,
            **({"profile": prof_s} if return_profile else {})
        }

    # Normalize profile to 0..1 based on min/max for numeric stability
    minv = float(prof_s.min())
    maxv = float(prof_s.max())
    rng = maxv - minv if (maxv - minv) > 1e-9 else 1.0
    prof_n = (prof_s - minv) / rng

    # Find local maxima (peaks) and minima (valleys) using simple neighbor comparison
    # Note: this is robust and avoids scipy dependency
    left = prof_n[:-2]
    center = prof_n[1:-1]
    right = prof_n[2:]
    # indices in original prof_n (offset by +1)
    peaks_idx = np.where((center > left) & (center >= right))[0] + 1
    valleys_idx = np.where((center < left) & (center <= right))[0] + 1

    # Optionally filter peaks by simple prominence: peak must be above neighbors by min_peak_prominence
    if min_peak_prominence is not None and min_peak_prominence > 0:
        good_peaks = []
        for p in peaks_idx:
            left_val = prof_n[p-1] if p-1 >= 0 else prof_n[p]
            right_val = prof_n[p+1] if p+1 < n else prof_n[p]
            local_prom = prof_n[p] - max(left_val, right_val)
            if local_prom >= min_peak_prominence:
                good_peaks.append(p)
        peaks_idx = np.array(good_peaks, dtype=int)

    # Recompute valleys that are strictly between peaks (optional)
    peaks_idx_sorted = np.sort(peaks_idx)
    valleys_idx_sorted = np.sort(valleys_idx)

    # For each valley, find nearest peak to left and right (must exist)
    deep_depths = []
    all_depths = []
    for v in valleys_idx_sorted:
        left_peaks = peaks_idx_sorted[peaks_idx_sorted < v]
        right_peaks = peaks_idx_sorted[peaks_idx_sorted > v]
        if left_peaks.size == 0 or right_peaks.size == 0:
            continue
        lp = left_peaks[-1]
        rp = right_peaks[0]
        peak_height = min(prof_n[lp], prof_n[rp])
        valley_val = prof_n[v]
        depth = peak_height - valley_val
        if depth < 0:
            depth = 0.0
        all_depths.append(depth)
        # deep if depth proportion of overall range >= depth_threshold_frac OR absolute min
        if depth >= max(depth_threshold_frac, abs_depth_min):
            deep_depths.append(depth)

    peaks_count = int(peaks_idx_sorted.size)
    valleys_count = int(valleys_idx_sorted.size)
    deep_count = int(len(deep_depths))
    avg_depth = float(np.mean(all_depths)) if len(all_depths) > 0 else 0.0
    avg_deep = float(np.mean(deep_depths)) if len(deep_depths) > 0 else 0.0

    deep_ratio = float(deep_count) / float(max(1, max(1, peaks_count)))  # denom safe

    out = {
        "peaks_count": peaks_count,
        "valleys_count": valleys_count,
        "deep_valleys_count": deep_count,
        "deep_valley_ratio": deep_ratio,
        "avg_valley_depth": avg_depth,
        "avg_deep_depth": avg_deep,
        "profile_range": rng,
    }
    if return_profile:
        out["profile"] = prof_s
        
    #visualize 
    return out

    # visual

# img_cell ="words/cell_r0_c2_blob_2_word_3.png"
img_cell ="blobs/cell_r8_c19_blob_1.png"


img = cv2.imread(img_cell)

res = count_peaks_and_deep_valleys(img, smooth_win=5,
                                   min_peak_prominence=0.02,
                                   depth_threshold_frac=0.35,
                                   abs_depth_min=0.03)

print(res)
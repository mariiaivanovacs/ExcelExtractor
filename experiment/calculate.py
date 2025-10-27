import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd

def centroid_of_image(image: np.ndarray,
                      mask: np.ndarray = None,
                      threshold: float = None,
                      normalize: bool = False,
                      visualize: bool = False,
                      return_int: bool = False):
    """
    Compute centroid (center of mass) of a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D array (H, W). Pixel values can be in 0..255 or 0..1 (dtype ignored).
    mask : np.ndarray, optional
        Boolean or numeric mask (same shape as image). If provided:
          - If boolean, only pixels where mask==True are considered (weight = image).
          - If numeric, it is used as an additional multiplicative weight.
    threshold : float, optional
        If provided, a boolean mask is created as (image > threshold) and applied
        (used when you want to ignore background automatically).
    normalize : bool, default False
        If True, the returned coordinates are normalized to [0,1] as (x/W, y/H).
    visualize : bool, default False
        If True, displays a plot of the image with centroid marked.
    return_int : bool, default False
        If True, also return integer-rounded coordinates (x_int, y_int) alongside floats.

    Returns
    -------
    (x_c, y_c) or (x_c, y_c, x_int, y_int)
        x_c = column coordinate (float), y_c = row coordinate (float).
        If normalize=True, these are in [0,1]. If return_int=True, returns integers too.
    """
    if image.ndim != 2:
        raise ValueError("image must be a 2D array (grayscale).")

    I = image.astype(float)
    H, W = I.shape
    
    df = pd.DataFrame(image)
    

    # Build effective weight map
    weights = I.copy()

    if threshold is not None:
        th_mask = (I > threshold).astype(float)
        weights *= th_mask

    
    import numpy as np

    mask = (df.to_numpy() <= 200).astype(float) 
    mask_arr = mask
    masked_image = I * mask_arr 
    s_y = np.sum(masked_image, axis=1) 
    s_x = np.sum(masked_image, axis=0) 
    binary_y = (s_y != 0).astype(int)
    
    binary_x = (s_x != 0).astype(int) 
    indexes_y = np.arange(-(H//2), H//2, 1) 
    indexes_x = np.arange(-(W//2), W//2, 1) 
    sum_y = np.sum(indexes_y * binary_y) 
    sum_x = np.sum(indexes_x * binary_x) 
    mass = np.sum(image)
    
    # image = image[0:10, 0:10]
    I = np.asarray(image, dtype=float)
    mask = (I < 255).astype(int)
    mass = mask.sum()
    # print(f"Mass: {mass}")
    
    
    central_x = sum_x / mass
    central_y = sum_y / mass
    

    
    
    
    
    # cv2.imwrite("experiment/masked_image.png", masked_image)
    if mask is not None:
        if mask.shape != I.shape:
            raise ValueError("mask must have same shape as image")
        mask_arr = mask.astype(float)
        weights *= mask_arr

    M00 = I.sum()  # total intensity (mass)

    if M00.item() == 0:
        # no mass: default to image center
        x_c = W / 2.0
        y_c = H / 2.0
    else:
        y_idx, x_idx = np.mgrid[0:H, 0:W]
        M10 = (x_idx * weights).sum()
        M01 = (y_idx * weights).sum()
        x_c = M10 / M00
        y_c = M01 / M00
        

    # Normalized coordinates if requested
    if normalize:
        x_c_norm = x_c / float(W)
        y_c_norm = y_c / float(H)

    # Visualization
    if visualize:
        fig, ax = plt.subplots(figsize=(5,5))

        # Display the image with custom extent
        ax.imshow(
            I,
            cmap='gray',
            interpolation='nearest',
            extent=[indexes_x[0], indexes_x[-1], indexes_y[-1], indexes_y[0]]  # match centered axes
        )

        # Mark centroid (note: convert from pixel coords to centered coords)
        # central_x_centered = central_x - (W / 2)
        # central_y_centered = central_y - (H / 2)

        ax.scatter(
            [x_c],
            [y_c],
            marker='x',
            s=80,
            linewidths=2,
            color='red'
        )

        # ax.invert_yaxis()  # show as image coordinates
        # plt.show()
        # save image
        fig.savefig("experiment/centroid.png")
    return central_x, central_y
    # if return_int:
    #     x_int = int(round(x_c))
    #     y_int = int(round(y_c))
    #     if normalize:
    #         return (x_c_norm, y_c_norm, x_int, y_int)
    #     else:
    #         return (x_c, y_c, x_int, y_int)
    # else:
    #     return (x_c_norm, y_c_norm) if normalize else (x_c, y_c)
    



import numpy as np
import matplotlib.pyplot as plt

def show_central_mass(I, threshold=0, ax=None, figsize=(5,5), cmap='gray', annotate=True):
    """
    Calculate binary mass (count of pixels > threshold), compute geometric centroid
    in centered coordinates, and plot image with centroid marker.
    
    Parameters
    ----------
    I : 2D array-like (grayscale)
        Input image.
    threshold : scalar
        Pixels > threshold are considered object (default 0).
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axes; otherwise create a new figure.
    figsize : tuple
        Figure size used when ax is None.
    cmap : str
        Colormap for imshow.
    annotate : bool
        Whether to show a text annotation with mass and centroid.
    
    Returns
    -------
    mass : int
        Number of object pixels (binary mass).
    centroid : tuple (x_c, y_c)
        Centroid in centered coordinates (x to the right, y upward). If no object pixels,
        returns (0.0, 0.0) and mass == 0.
    """
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise ValueError("I must be a 2D grayscale image.")
    H, W = I.shape
    

    # centered coordinate arrays: map pixel index i -> coordinate (i - (N-1)/2)
    xs = np.arange(W) - (W - 1) / 2.0
    ys = np.arange(H) - (H - 1) / 2.0

    # binary mask and mass
    mask = (I < 230)
    mass = int(mask.sum())

    if mass == 0:
        print("NO OBJECT PIXELS")
        # no object pixels: centroid fallback to image center in centered coords -> (0,0)
        x_c = 0.0
        y_c = 0.0
    else:
        ys_idx, xs_idx = np.nonzero(mask)            # row indices (y), column indices (x)
        x_c = xs[xs_idx].mean()
        y_c = ys[ys_idx].mean()

    # plotting
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # extent so that pixels map to centered coordinates and origin='lower' for y-upwards
    extent = [xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5]
    ax.imshow(I, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')
    ax.scatter([x_c], [y_c], marker='x', s=100, linewidths=2, color='red', label='centroid')

    # axes lines at zero
    ax.axhline(0.0, color='cyan', linestyle='--', linewidth=1)
    ax.axvline(0.0, color='cyan', linestyle='--', linewidth=1)

    # labels & ticks: set ticks at integer coordinates if possible
    x_min, x_max = xs[0], xs[-1]
    y_min, y_max = ys[0], ys[-1]
    # choose integer ticks spanning the range
    xticks = np.arange(np.ceil(x_min), np.floor(x_max) + 1, 1)
    yticks = np.arange(np.ceil(y_min), np.floor(y_max) + 1, 1)
    if len(xticks) <= 15:  # avoid too many ticks
        ax.set_xticks(xticks)
    if len(yticks) <= 15:
        ax.set_yticks(yticks)

    ax.set_xlabel("x (centered)")
    ax.set_ylabel("y (centered)")

    if annotate:
        if mass == 0:
            txt = "Mass=0 (empty). Centroid=(0.00, 0.00)"
        else:
            txt = f"Mass={mass}, Centroid=(x={x_c:.4f}, y={y_c:.4f})"
        # place annotation top-left inside the axes
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_title("Image (centered coords) with centroid")
    ax.set_aspect('equal')

    random_int = random.randint(0, 1000000)
    # plt.savefig(f"experiment/test_{random_int}.png")
    # print(f"Mass: {mass}, Centroid: {(x_c, y_c)}")

    return mass, (x_c, y_c)

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from typing import Tuple, List, Dict, Any

def find_best_symmetry_center(I,
                              upsample: int = 1,
                              search_full: bool = True,
                              search_radius: int | None = None) -> Tuple[Tuple[int,int], float]:
    """
    Find the integer (cx,cy) center (in the upsampled image coordinate system)
    that minimizes the 180-degree symmetry error.
    Returns (cx, cy), min_error.
    """
    Iu = I if upsample == 1 else np.kron(I, np.ones((upsample, upsample)))
    H, W = Iu.shape

    # candidate centers: if search_full then all integer coords in image,
    # otherwise search in a square around image center with radius search_radius.
    if not search_full:
        if search_radius is None:
            search_radius = min(H, W) // 4
        cx0 = (W - 1) // 2
        cy0 = (H - 1) // 2
        cx_range = range(max(0, cx0 - search_radius), min(W, cx0 + search_radius + 1))
        cy_range = range(max(0, cy0 - search_radius), min(H, cy0 + search_radius + 1))
    else:
        cx_range = range(0, W)
        cy_range = range(0, H)

    # precompute indices grid
    ii, jj = np.indices((H, W))  # ii: rows (y), jj: cols (x)

    best_error = np.inf
    best_center = ( (W-1)//2, (H-1)//2 )

    for cy in cy_range:
        for cx in cx_range:
            # symmetric coordinates: si = 2*cy - i, sj = 2*cx - j
            si = 2 * cy - ii
            sj = 2 * cx - jj

            valid = (si >= 0) & (si < H) & (sj >= 0) & (sj < W)
            if not np.any(valid):
                continue

            A = Iu[ii[valid], jj[valid]]
            B = Iu[si[valid], sj[valid]]

            # half to avoid double-counting pairs
            err = 0.5 * np.sum((A - B) ** 2)

            # optionally normalize by number of valid pairs:
            #err_norm = err / np.count_nonzero(valid)

            if err < best_error:
                best_error = err
                best_center = (cx, cy)

    return best_center, float(best_error)


def compute_image_moments(I: np.ndarray) -> Dict[str, float]:
    """
    Compute image intensity moments using centered coordinates:
      - zeroth: m00 = sum(I)
      - first raw moments: m10, m01
      - second raw moments: m20, m02, m11
      - centroid: x_c = m10/m00, y_c = m01/m00  (if m00>0)
      - central second moments: mu20, mu02, mu11
    Coordinates: x axis -> columns, y axis -> rows.
    Origin for x,y is the image center: x = col - (W-1)/2, y = row - (H-1)/2
    Returns dict of floats.
    """
    I = np.asarray(I, dtype=float)
    H, W = I.shape
    xs = np.arange(W) - (W - 1) / 2.0
    ys = np.arange(H) - (H - 1) / 2.0
    jj, ii = np.meshgrid(xs, ys)   # jj -> x coords grid, ii -> y coords grid

    m00 = float(np.sum(I))
    m10 = float(np.sum(I * jj))
    m01 = float(np.sum(I * ii))
    m20 = float(np.sum(I * (jj ** 2)))
    m02 = float(np.sum(I * (ii ** 2)))
    m11 = float(np.sum(I * (jj * ii)))

    moments = {
        'm00': m00, 'm10': m10, 'm01': m01, 'm20': m20, 'm02': m02, 'm11': m11
    }

    if m00 > 0:
        x_c = m10 / m00
        y_c = m01 / m00
    else:
        x_c = 0.0
        y_c = 0.0

    # central second moments
    mu20 = float(np.sum(I * ((jj - x_c) ** 2)))
    mu02 = float(np.sum(I * ((ii - y_c) ** 2)))
    mu11 = float(np.sum(I * ((jj - x_c) * (ii - y_c))))

    moments.update({
        'x_centroid': float(x_c),
        'y_centroid': float(y_c),
        'mu20': mu20, 'mu02': mu02, 'mu11': mu11
    })

    return moments


def show_symmetry_and_store(I,
                            output_csv: str = "image_moments.csv",
                            image_id: str = "img",
                            upsample: int = 1,
                            search_full: bool = True,
                            search_radius: int | None = None,
                            min_intensity: float | None = None,
                            plot: bool = True,
                            annotate: bool = True) -> Dict[str, Any]:
    """
    Search for best point-of-symmetry (integer grid, optionally upsampled),
    compute first & second moments (raw and central), optionally plot image
    showing symmetry center, and append results to CSV.

    Args:
        I: 2D grayscale image (numpy array)
        output_csv: path to CSV file where results are appended
        image_id: identifier string for this image (filename or index)
        upsample: integer factor to upsample image for subpixel search
        search_full: whether to search the full upsampled image for best center
        search_radius: if search_full is False, radius around image center to search
        min_intensity: if given, clip image to max intensity min_intensity (useful to handle background)
        plot: whether to create and show/save a plot (returns data regardless)
        annotate: whether to annotate plot with computed numbers

    Returns:
        result dict containing moments, best symmetry center (in original image coords),
        symmetry error, and path to CSV entry.
    """
    I = np.asarray(I, dtype=float)
    if I.ndim != 2:
        raise ValueError("I must be a 2D grayscale image.")

    # optional intensity clamp (helps if darker=object but background varies)
    if min_intensity is not None:
        I = np.clip(I, None, float(min_intensity))

    H, W = I.shape

    # Find best symmetry center in upsampled coordinates
    best_center_u, best_error = find_best_symmetry_center(I, upsample=upsample,
                                                          search_full=search_full,
                                                          search_radius=search_radius)

    # convert best center back to original image coordinates (float)
    cx_u, cy_u = best_center_u
    cx = cx_u / upsample
    cy = cy_u / upsample

    # compute moments on original image
    moments = compute_image_moments(I)

    # prepare result dictionary
    result = {
        'image_id': str(image_id),
        'image_width': int(W),
        'image_height': int(H),
        'best_sym_center_x': float(cx),
        'best_sym_center_y': float(cy),
        'best_sym_error': float(best_error),
    }
    result.update(moments)

    # append to CSV (create header if new)
    header = ['image_id', 'image_width', 'image_height',
              'best_sym_center_x', 'best_sym_center_y', 'best_sym_error',
              'm00', 'm10', 'm01', 'm20', 'm02', 'm11',
              'x_centroid', 'y_centroid', 'mu20', 'mu02', 'mu11']

    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode='a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        # create row dict mapping header names to result values
        row = {k: result.get(k, '') for k in header}
        writer.writerow(row)

    # plotting (optional)
    if plot:
        xs = np.arange(W) - (W - 1) / 2.0
        ys = np.arange(H) - (H - 1) / 2.0
        extent = [xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(I, cmap='gray', interpolation='nearest', extent=extent, origin='lower')

        # plot best symmetry center (in centered coords)
        cx_centered = cx - (W - 1) / 2.0
        cy_centered = cy - (H - 1) / 2.0
        ax.scatter([cx_centered], [cy_centered], marker='x', s=120, linewidths=2, color='red', label='sym_center')

        # mark centroid too (from moments) for comparison
        ax.scatter([moments['x_centroid']], [moments['y_centroid']], marker='o', s=80, facecolors='none', edgecolors='blue', label='intensity_centroid')

        ax.axhline(0.0, color='cyan', linestyle='--', linewidth=1)
        ax.axvline(0.0, color='cyan', linestyle='--', linewidth=1)

        if annotate:
            txt = (f"sym_center=(x={cx:.3f}, y={cy:.3f}), err={best_error:.1f}\n"
                   f"m00={moments['m00']:.2f}, centroid=(x={moments['x_centroid']:.3f}, y={moments['y_centroid']:.3f})\n"
                   f"mu20={moments['mu20']:.2f}, mu02={moments['mu02']:.2f}")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), fontsize=9)

        ax.set_xlabel("x (centered)")
        ax.set_ylabel("y (centered)")
        ax.set_title(f"Image {image_id}: symmetry center vs intensity centroid")
        ax.set_aspect('equal')
        ax.legend(loc='lower right')
        plt.tight_layout()
        # do not automatically save — user can save or display outside
        plt.show()
        plt.close(fig)

    return result


# img = cv2.imread("characters/cell_r8_c15_blob_1_word_2_char_01.png", cv2.IMREAD_GRAYSCALE)
# # single image (2D numpy array)
# res = show_symmetry_and_store(img, output_csv='experiment/mymoments.csv', image_id='img001', upsample=2, plot=True)



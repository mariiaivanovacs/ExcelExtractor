from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from scipy.stats import pearsonr

def visualize_feature_pair_correlations(
    df: pd.DataFrame,
    feature_pairs: Optional[List[Tuple[str, str]]] = None,
    group_col: Optional[str] = None,
    top_n: int = 12,
    method: str = "pearson",         # correlation method for selection (pearson/spearman)
    n_cols: int = 3,
    point_size: int = 30,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    show_heatmap: bool = False,
    heatmap_diag: bool = False,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize correlation / scatter for feature pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with numeric features and optional group column.
    feature_pairs : Optional[list of (feat1, feat2)]
        Explicit list of feature pairs to plot. If None, function will auto-select top_n pairs
        ranked by absolute correlation (method controls pearson/spearman).
    group_col : Optional[str]
        Column name to color points by (e.g. 'label' or 'character'). If None, no grouping color used.
    top_n : int
        Number of top correlated pairs to choose when feature_pairs is None.
    method : str
        'pearson' or 'spearman' used to compute correlation for auto-selection.
    n_cols : int
        Number of columns in the subplot grid.
    point_size : int
        Scatter point size.
    figsize_per_plot : (w, h)
        Size per subplot; total figsize is scaled by number of rows.
    show_heatmap : bool
        If True, also plot a small correlation heatmap of all numeric features used.
    heatmap_diag : bool
        If True, show diagonal values in heatmap (otherwise masked).
    save_path : Optional[str]
        If provided, save the figure to this path.
    show : bool
        If False, the function returns the matplotlib Figure without calling plt.show().
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing scatter subplots (and heatmap if requested).
    """
    # --- validate inputs ---
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    # Select numeric columns only
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns found in DataFrame.")
    
    # If user gave feature_pairs, validate them
    if feature_pairs is not None:
        # filter-out invalid column names
        feature_pairs = [(a, b) for (a, b) in feature_pairs if a in df.columns and b in df.columns]
        if not feature_pairs:
            raise ValueError("No valid feature pairs found among provided names.")
    else:
        # compute correlation matrix and pick top_n absolute correlations from upper triangle
        corr = df[numeric].corr(method=method).abs()
        # upper triangle excluding diagonal
        tri_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        flat = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                flat.append((cols[i], cols[j], corr.iloc[i, j]))
        flat_sorted = sorted(flat, key=lambda x: x[2], reverse=True)
        selected = flat_sorted[:top_n]
        feature_pairs = [(a, b) for a, b, _ in selected]
        if len(feature_pairs) == 0:
            raise ValueError("Could not select any correlated pairs (maybe too few numeric cols).")
    
    # Prepare plotting grid
    n_pairs = len(feature_pairs)
    n_rows = ceil(n_pairs / n_cols)
    fig_w = n_cols * figsize_per_plot[0]
    fig_h = n_rows * figsize_per_plot[1]
    
    # If heatmap requested, make space above subplots
    if show_heatmap:
        fig = plt.figure(figsize=(fig_w, fig_h + 4))
        gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.8] + [figsize_per_plot[1]]*n_rows)
        heatmap_ax = fig.add_subplot(gs[0, :])
        # subplots start at row 1
        sub_axes = []
        for r in range(n_rows):
            for c in range(n_cols):
                sub_axes.append(fig.add_subplot(gs[r+1, c]))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
        sub_axes = axes.flatten()
        heatmap_ax = None
    
    palette = None
    if group_col is not None and group_col in df.columns:
        groups = df[group_col].astype(str).unique()
        palette = dict(zip(sorted(groups), sns.color_palette(n_colors=len(groups))))
    
    # --- Optionally plot heatmap of correlations for numeric features ---
    if show_heatmap and heatmap_ax is not None:
        corr_all = df[numeric].corr(method=method)
        mask = None
        if not heatmap_diag:
            mask = np.triu(np.ones_like(corr_all, dtype=bool))
        sns.heatmap(corr_all, ax=heatmap_ax, cmap="vlag", center=0, annot=False, mask=mask)
        heatmap_ax.set_title(f"Correlation matrix ({method}) for numeric features")
    
    # --- Plot each pair ---
    for i, (feat1, feat2) in enumerate(feature_pairs):
        if i >= len(sub_axes):
            break
        ax = sub_axes[i]
        
        # check existence
        if feat1 not in df.columns or feat2 not in df.columns:
            ax.text(0.5, 0.5, f"Missing: {feat1} or {feat2}", ha="center")
            ax.set_axis_off()
            continue
        
        # drop NA rows for those columns + optional group_col
        cols_to_dropna = [feat1, feat2] + ([group_col] if (group_col and group_col in df.columns) else [])
        sub = df[cols_to_dropna].dropna()
        if sub.shape[0] == 0:
            ax.text(0.5, 0.5, f"No data for {feat1} vs {feat2}", ha="center")
            ax.set_axis_off()
            continue
        
        # If grouping specified, scatter per group
        if group_col and group_col in df.columns:
            for g, gdf in sub.groupby(group_col):
                ax.scatter(gdf[feat1], gdf[feat2], label=str(g), alpha=0.7, s=point_size,
                           c=[palette[str(g)]] if palette else None)
        else:
            ax.scatter(sub[feat1], sub[feat2], alpha=0.7, s=point_size)
        
        # Compute and annotate overall Pearson r (or Spearman if selected)
        try:
            if method == "pearson":
                r, p = pearsonr(sub[feat1], sub[feat2])
            else:
                from scipy.stats import spearmanr
                r, p = spearmanr(sub[feat1], sub[feat2])
            r_text = f"r={r:.3f}, p={p:.2e}"
        except Exception:
            r_text = "r=nan"
        # place annotation in top-left of axes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(0.02*(xmax-xmin)+xmin, 0.95*(ymax-ymin)+ymin, r_text, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.7, facecolor="white"))
        
        ax.set_xlabel(feat1)
        ax.set_ylabel(feat2)
        ax.set_title(f"{feat1} vs {feat2}")
        ax.grid(alpha=0.25)
        if group_col and group_col in df.columns:
            ax.legend(fontsize=8, markerscale=0.6, loc="best")
    
    # turn off any extra axes
    for j in range(n_pairs, len(sub_axes)):
        sub_axes[j].axis("off")
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved correlation pairs plot to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_feature_distributions(
    df: pd.DataFrame, 
    class_col: str = 'character',
    save_path: str = 'results/feature_distributions.png'
):
    """Visualize feature distributions for each class (handles constant or narrow ranges safely)."""
    
    features = [col for col in df.select_dtypes(include=['number']).columns if col != class_col]
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    class_values = sorted(df[class_col].astype(str).unique())
    
    for i, feature in enumerate(features):
        ax = axes[i]
        data_all = df[feature].dropna()
        
        # Skip constant or empty features
        if data_all.nunique() < 2:
            ax.text(0.5, 0.5, "Constant", ha='center', va='center')
            ax.set_title(feature)
            ax.axis('off')
            continue
        
        # Choose bins dynamically based on data range
        n_bins = min(20, max(5, data_all.nunique()))
        
        for label in class_values:
            data = df.loc[df[class_col].astype(str) == label, feature].dropna()
            if len(data) == 0:
                continue
            ax.hist(data, alpha=0.5, label=label, bins=n_bins)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution: {feature}')
        ax.legend(fontsize=8, ncol=5)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature distributions saved to: {save_path}")
    plt.close()





df = pd.read_csv("experiment/only_digits.csv")
# visualize_feature_pair_correlations(df)
visualize_feature_distributions(df)


# import joblib
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load model
# model = joblib.load("models/RandomForest.pkl")

# # Load your training data (same features order)
# df = pd.read_csv("experiment/only_digits.csv")
# X = df.drop(["character", "variant", "center_x", "center_y"], axis=1)
# feature_names = X.columns

# # Get feature importances
# importances = model.feature_importances_
# # print("importances:", importances)

# # Sort by importance
# indices = np.argsort(importances)[::-1]
# top_n = 20  # number of top features to show

# print("Top features by importance:")
# features = []
# for i in range(top_n):
#     features.append(feature_names[indices[i]])
#     print(f"{i+1:2d}. {feature_names[indices[i]]:30s} {importances[indices[i]]:.4f}")

# print("features:", features)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.barh(range(top_n), importances[indices[:top_n]][::-1], align="center")
# plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
# plt.xlabel("Feature Importance")
# plt.title("Top 20 Most Informative Features (RandomForest)")
# plt.tight_layout()
# # plt.show()

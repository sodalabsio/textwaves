from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Fixed label order for consistent colors across methods/datasets
LABEL_ORDER: List[str] = ['deep_democracy', 'mainstream_democracy', 'anti_democratic']

# Define a stable, readable color map (Set2 in a fixed order)
_DEFAULT_PALETTE = sns.color_palette('Set2', n_colors=3)
_COLOR_MAP: Dict[str, tuple] = {
    'deep_democracy': _DEFAULT_PALETTE[0],    # teal-ish
    'mainstream_democracy': _DEFAULT_PALETTE[1],  # lavender
    'anti_democratic': _DEFAULT_PALETTE[2],   # orange
}

# Marker shapes by cluster labels (cycled if more than available)
_MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h']


def scatter_2d(coords: np.ndarray, df: pd.DataFrame, cluster_labels: np.ndarray,
               title: str, path: str):
    """2D scatter:
    - Color encodes true labels (fixed mapping for consistency across runs).
    - Marker shape encodes discovered clusters.
    Saves figure to `path`.
    """
    plt.figure(figsize=(6.4, 5.2))
    sns.set_style('whitegrid')
    ax = plt.gca()

    # Ensure label column exists and map to fixed colors
    if 'label_name' not in df.columns:
        raise ValueError("df must contain a 'label_name' column with true labels")

    # Build color array with a per-element default (avoid fillna with tuple)
    def _color_for(lbl: str):
        return _COLOR_MAP.get(str(lbl), (0.5, 0.5, 0.5))

    colors = [_color_for(lbl) for lbl in df['label_name']]

    # Cluster markers
    clus = pd.Series(cluster_labels).astype('category')
    cat_clusters = list(clus.cat.categories)
    marker_map = {cat: _MARKERS[i % len(_MARKERS)] for i, cat in enumerate(cat_clusters)}

    # Plot by cluster so shapes are consistent in legend
    for c_lab in cat_clusters:
        idx = (clus == c_lab).values
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=np.array(colors, dtype=object)[idx].tolist(),
            s=32, marker=marker_map[c_lab], edgecolors='black', linewidths=0.3,
            alpha=0.9, label=f'cluster {c_lab}',
        )

    # Legends: one for clusters (shapes) and one for true labels (colors)
    # Cluster legend handles (uniform color with distinct markers)
    cluster_handles = [
        Line2D([0], [0], marker=marker_map[c], color='w', markerfacecolor='#999999',
               markeredgecolor='black', markersize=7, linewidth=0, label=f'cluster {c}')
        for c in cat_clusters
    ]
    cluster_legend = ax.legend(handles=cluster_handles, loc='lower right', fontsize=8, frameon=True, title='Clusters')
    ax.add_artist(cluster_legend)

    # Label legend handles (distinct colors with a default marker)
    present_labels = [lbl for lbl in LABEL_ORDER if lbl in set(df['label_name'])]
    # Include any unforeseen labels as well
    extras = [lbl for lbl in sorted(set(df['label_name'])) if lbl not in present_labels]
    ordered_labels = present_labels + extras
    label_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_color_for(lbl),
               markeredgecolor='black', markersize=7, linewidth=0, label=lbl)
        for lbl in ordered_labels
    ]
    ax.legend(handles=label_handles, loc='upper left', fontsize=8, frameon=True, title='True labels')

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

from typing import Dict, List, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Preferred legacy order retained when those labels are present, but plotting now
# adapts to any observed label set.
PREFERRED_LABEL_ORDER: List[str] = [
    'deep_democracy',
    'mainstream_democracy',
    'anti_democratic',
]

# Marker shapes reused for clusters and, in the true-only view, for labels.
_MARKERS = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h']


def ordered_labels(labels: Iterable[object]) -> List[str]:
    present = sorted({str(x).strip() for x in labels if pd.notna(x) and str(x).strip()})
    preferred = [lbl for lbl in PREFERRED_LABEL_ORDER if lbl in present]
    extras = [lbl for lbl in present if lbl not in preferred]
    return preferred + extras


def color_map_for_labels(labels: Iterable[object]) -> Dict[str, tuple]:
    ordered = ordered_labels(labels)
    if not ordered:
        return {}
    # Use a stable palette; legacy labels keep the first three colours when present.
    palette = sns.color_palette('Set2', n_colors=max(3, len(ordered)))
    return {lbl: palette[i] for i, lbl in enumerate(ordered)}


def _marker_map_for_items(items: Iterable[object]) -> Dict[str, str]:
    ordered = ordered_labels(items)
    return {lbl: _MARKERS[i % len(_MARKERS)] for i, lbl in enumerate(ordered)}


def scatter_2d(coords: np.ndarray, df: pd.DataFrame, cluster_labels: np.ndarray,
               title: str, path: str):
    """2D scatter:
    - Color encodes true labels (derived from the observed label set).
    - Marker shape encodes discovered clusters.
    Saves figure to `path`.
    """
    plt.figure(figsize=(6.4, 5.2))
    sns.set_style('whitegrid')
    ax = plt.gca()

    if 'label_name' not in df.columns:
        raise ValueError("df must contain a 'label_name' column with true labels")

    label_order = ordered_labels(df['label_name'])
    color_map = color_map_for_labels(df['label_name'])

    def _color_for(lbl: str):
        return color_map.get(str(lbl), (0.5, 0.5, 0.5))

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
    cluster_handles = [
        Line2D([0], [0], marker=marker_map[c], color='w', markerfacecolor='#999999',
               markeredgecolor='black', markersize=7, linewidth=0, label=f'cluster {c}')
        for c in cat_clusters
    ]
    cluster_legend = ax.legend(handles=cluster_handles, loc='lower right', fontsize=8, frameon=True, title='Clusters')
    ax.add_artist(cluster_legend)

    label_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_color_for(lbl),
               markeredgecolor='black', markersize=7, linewidth=0, label=lbl)
        for lbl in label_order
    ]
    ax.legend(handles=label_handles, loc='upper left', fontsize=8, frameon=True, title='True labels')

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def scatter_true_only_2d(coords: np.ndarray, df: pd.DataFrame, title: str, path: str):
    """2D scatter using only true labels.

    Both colour and marker shape encode the true label, and no cluster information is
    used at all. This is useful as a simple visual check of apparent separation in the
    reduced representational space.
    """
    plt.figure(figsize=(6.4, 5.2))
    sns.set_style('whitegrid')
    ax = plt.gca()

    if 'label_name' not in df.columns:
        raise ValueError("df must contain a 'label_name' column with true labels")

    label_order = ordered_labels(df['label_name'])
    color_map = color_map_for_labels(df['label_name'])
    marker_map = _marker_map_for_items(df['label_name'])

    for lbl in label_order:
        idx = (df['label_name'].astype(str) == str(lbl)).values
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=[color_map[lbl]],
            s=34,
            marker=marker_map[lbl],
            edgecolors='black',
            linewidths=0.35,
            alpha=0.9,
            label=lbl,
        )

    label_handles = [
        Line2D([0], [0], marker=marker_map[lbl], color='w', markerfacecolor=color_map[lbl],
               markeredgecolor='black', markersize=7, linewidth=0, label=lbl)
        for lbl in label_order
    ]
    ax.legend(handles=label_handles, loc='upper left', fontsize=8, frameon=True, title='True labels')

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

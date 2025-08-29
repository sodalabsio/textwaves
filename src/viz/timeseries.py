import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Iterable, Dict

# Reuse consistent colors from scatter if available
try:
    from .scatter import _COLOR_MAP as _SCATTER_COLOR_MAP, LABEL_ORDER as _LABEL_ORDER
except Exception:
    _SCATTER_COLOR_MAP = None  # type: ignore
    _LABEL_ORDER = []  # type: ignore


def _color_for(lbl: str):
    if _SCATTER_COLOR_MAP is not None:
        return _SCATTER_COLOR_MAP.get(lbl, (0.5, 0.5, 0.5))
    # Fallback palette
    palette = sns.color_palette('Set2', n_colors=6)
    return palette[hash(lbl) % len(palette)]


def _ordered_labels(present: Iterable[str]) -> list[str]:
    present = list(sorted(set(present)))
    if _LABEL_ORDER:
        ordered = [x for x in _LABEL_ORDER if x in present]
        extras = [x for x in present if x not in ordered]
        return ordered + extras
    return present


def plot_monthly_label_proportions(df: pd.DataFrame, path: str, cutoff_timestamp: Optional[pd.Timestamp] = None):
    d = df.copy()
    if not isinstance(d['timestamp'].iloc[0], pd.Timestamp):
        d['timestamp'] = pd.to_datetime(d['timestamp'])
    d['month'] = d['timestamp'].dt.to_period('M').dt.to_timestamp()
    grp = d.groupby(['month', 'label_name']).size().reset_index(name='n')
    total = grp.groupby('month')['n'].transform('sum')
    grp['share'] = grp['n'] / total

    plt.figure(figsize=(7.6, 4.4))
    ax = plt.gca()

    # Use consistent colors if available
    if _SCATTER_COLOR_MAP is not None:
        pal: Dict[str, tuple] = {lbl: _color_for(lbl) for lbl in _ordered_labels(grp['label_name'])}
        sns.lineplot(data=grp, x='month', y='share', hue='label_name', marker='o', palette=pal, ax=ax)
    else:
        sns.lineplot(data=grp, x='month', y='share', hue='label_name', marker='o', ax=ax)

    # Optional vertical cutoff marker
    if cutoff_timestamp is not None:
        # Draw a vertical line at the cutoff time (start of test period)
        ax.axvline(cutoff_timestamp, color='black', linestyle='--', linewidth=1.2, alpha=0.7, zorder=5)
        # Light annotation
        ylim = ax.get_ylim()
        y_text = min(ylim[1], 0.98)
        ax.text(cutoff_timestamp, y_text, ' train/test split ', rotation=90,
                va='top', ha='left', fontsize=7, color='black', alpha=0.8,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.6))

    ax.set_ylabel('Proportion', fontsize=9)
    ax.set_xlabel('Month', fontsize=9)
    ax.set_title('Monthly label proportions', fontsize=11)
    ax.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_future_reconstruction(df: pd.DataFrame, test_idx: pd.Index, y_pred, path: str,
                               title: Optional[str] = None):
    """Overlay, for the FUTURE (test period), the monthly class shares from
    ground-truth vs. logistic-regression predictions.

    Styling:
      - Ground truth: thick, very transparent (alpha≈0.3), drawn underneath.
      - Predicted: thin, solid, higher alpha on top (no dashed lines).

    - df: full dataframe with columns ['timestamp', 'label', 'label_name']
    - test_idx: index labels corresponding to the future subset
    - y_pred: array-like of predicted integer labels, aligned with test_idx order
    - path: output path for the figure
    """
    dft = df.copy()
    if not isinstance(dft['timestamp'].iloc[0], pd.Timestamp):
        dft['timestamp'] = pd.to_datetime(dft['timestamp'])

    # Slice future
    fut = dft.loc[test_idx].copy()

    # Map predicted ints -> names using observed mapping in df
    label_map = dict(dft[['label', 'label_name']].drop_duplicates().values)
    fut['pred_label_name'] = [label_map.get(int(x), str(int(x))) for x in y_pred]

    # Month bins
    fut['month'] = fut['timestamp'].dt.to_period('M').dt.to_timestamp()

    # Ground-truth shares
    g = fut.groupby(['month', 'label_name']).size().reset_index(name='n')
    g['share'] = g.groupby('month')['n'].transform(lambda x: x / x.sum())

    # Predicted shares
    p = fut.groupby(['month', 'pred_label_name']).size().reset_index(name='n')
    p['share'] = p.groupby('month')['n'].transform(lambda x: x / x.sum())
    p = p.rename(columns={'pred_label_name': 'label_name'})

    # Ensure consistent ordering
    labels = _ordered_labels(list(g['label_name']) + list(p['label_name']))
    months = sorted(set(fut['month']))

    plt.figure(figsize=(7.6, 4.4))
    sns.set_style('whitegrid')
    ax = plt.gca()

    for lbl in labels:
        g_lbl = g[g['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})
        p_lbl = p[p['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})
        color = _color_for(lbl)
        # Ground-truth: thick, very transparent, underneath
        ax.plot(months, g_lbl['share'].values, '-', color=color, linewidth=3.0, alpha=0.3, zorder=1,
                label=f'{lbl} • ground truth')
        # Predicted: thin, solid, on top
        ax.plot(months, p_lbl['share'].values, '-', color=color, linewidth=1.4, alpha=0.95, zorder=3,
                label=f'{lbl} • predicted')

    ax.set_ylabel('Proportion', fontsize=9)
    ax.set_xlabel('Month (future only)', fontsize=9)
    ax.set_title(title or 'Future reconstruction: ground truth vs logistic regression (temporal split)', fontsize=11)
    ax.tick_params(axis='both', labelsize=8)
    # Build a single legend with compact entries and no duplicates
    handles, labels_txt = ax.get_legend_handles_labels()
    seen = set()
    handles_out = []
    labels_out = []
    for h, t in zip(handles, labels_txt):
        if t not in seen:
            handles_out.append(h)
            labels_out.append(t)
            seen.add(t)
    ax.legend(handles_out, labels_out, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_future_reconstruction_by_class(
    df: pd.DataFrame,
    test_idx: pd.Index,
    y_pred,
    path: str,
    title: Optional[str] = None,
    ncols: Optional[int] = None,
    sharey: bool = True,
    ylim: tuple[float, float] = (0.0, 1.0),
):
    """Faceted version of future reconstruction: one subplot per class (left-to-right).

    Each subplot shows, for the FUTURE (test period), the monthly class shares from
    ground-truth vs predicted, using the same styling as the overlay version.

    - df: full dataframe with columns ['timestamp', 'label', 'label_name']
    - test_idx: index labels corresponding to the future subset
    - y_pred: array-like of predicted integer labels, aligned with test_idx order
    - path: output path for the figure
    - title: optional suptitle for the whole figure
    - ncols: number of columns (defaults to one row: ncols == n_classes). If you have many classes,
             set ncols to a smaller number to wrap to multiple rows.
    - sharey: whether to share the y-axis across subplots (recommended for comparability)
    - ylim: y-axis limits for proportions
    """
    dft = df.copy()
    if not isinstance(dft['timestamp'].iloc[0], pd.Timestamp):
        dft['timestamp'] = pd.to_datetime(dft['timestamp'])

    # Slice future
    fut = dft.loc[test_idx].copy()

    # Map predicted ints -> names using observed mapping in df
    label_map = dict(dft[['label', 'label_name']].drop_duplicates().values)
    fut['pred_label_name'] = [label_map.get(int(x), str(int(x))) for x in y_pred]

    # Month bins
    fut['month'] = fut['timestamp'].dt.to_period('M').dt.to_timestamp()

    # Ground-truth shares
    g = fut.groupby(['month', 'label_name']).size().reset_index(name='n')
    if len(g) == 0:
        raise ValueError('No ground-truth observations in the provided test period (test_idx).')
    g['share'] = g.groupby('month')['n'].transform(lambda x: x / x.sum())

    # Predicted shares
    p = fut.groupby(['month', 'pred_label_name']).size().reset_index(name='n')
    p['share'] = p.groupby('month')['n'].transform(lambda x: x / x.sum())
    p = p.rename(columns={'pred_label_name': 'label_name'})

    # Ensure consistent ordering
    labels = _ordered_labels(list(g['label_name']) + list(p['label_name']))
    months = sorted(set(fut['month']))

    n_classes = len(labels)
    if ncols is None or ncols <= 0:
        ncols = n_classes  # one row by default (left-to-right)
    nrows = math.ceil(n_classes / ncols)

    # Size heuristics: ~2.8 inches per column, ~2.6 inches per row
    fig_w = max(6.0, 2.8 * ncols)
    fig_h = max(2.8, 2.6 * nrows)

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=sharey,
                             figsize=(fig_w, fig_h))

    # Normalize axes handling to a flat list
    if isinstance(axes, plt.Axes):  # single axis when nrows=ncols=1
        axes_list = [axes]
    else:
        axes_list = [ax for ax in axes.ravel()]

    for i, lbl in enumerate(labels):
        ax = axes_list[i]
        color = _color_for(lbl)

        g_lbl = g[g['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})
        p_lbl = p[p['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})

        # Ground-truth: thick, very transparent, underneath
        ax.plot(months, g_lbl['share'].values, '-', color=color, linewidth=3.0, alpha=0.3, zorder=1,
                label='ground truth')
        # Predicted: thin, solid, on top
        ax.plot(months, p_lbl['share'].values, '-', color=color, linewidth=1.4, alpha=0.95, zorder=3,
                label='predicted')

        ax.set_title(lbl, fontsize=9)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Only show legend on the first subplot to avoid clutter
        if i == 0:
            ax.legend(frameon=True, fontsize=8, loc='upper left')

        # Cosmetic: lighter x tick labels for non-bottom rows
        if (i // ncols) < (nrows - 1):
            ax.tick_params(axis='x', labelbottom=False)
        # Smaller tick labels overall
        ax.tick_params(axis='both', labelsize=8)

    # If there are empty axes (when grid > n_classes), hide them
    for j in range(len(axes_list)):
        if j >= n_classes:
            axes_list[j].axis('off')

    # Shared labels
    fig.suptitle(title or 'Future reconstruction by class: ground truth vs prediction', fontsize=11)
    fig.supxlabel('Month (future only)', fontsize=9)
    fig.supylabel('Proportion', fontsize=9)

    # Rotate x tick labels on the bottom row for readability
    for i in range((nrows - 1) * ncols, nrows * ncols):
        if i < len(axes_list):
            for label in axes_list[i].get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)

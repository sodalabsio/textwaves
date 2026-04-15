import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

# Reuse the same dynamic ordering and colour logic as scatter when available.
try:
    from .scatter import color_map_for_labels, ordered_labels
except Exception:
    def ordered_labels(labels):
        return sorted({str(x).strip() for x in labels if pd.notna(x) and str(x).strip()})

    def color_map_for_labels(labels):
        ordered = ordered_labels(labels)
        palette = sns.color_palette('Set2', n_colors=max(3, len(ordered) or 1))
        return {lbl: palette[i] for i, lbl in enumerate(ordered)}


def _color_for(lbl: str, palette_map):
    return palette_map.get(lbl, (0.5, 0.5, 0.5))


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

    label_order = ordered_labels(grp['label_name'])
    pal = color_map_for_labels(grp['label_name'])
    sns.lineplot(
        data=grp,
        x='month',
        y='share',
        hue='label_name',
        hue_order=label_order,
        marker='o',
        palette=pal,
        ax=ax,
    )

    # Optional vertical cutoff marker
    if cutoff_timestamp is not None:
        ax.axvline(cutoff_timestamp, color='black', linestyle='--', linewidth=1.2, alpha=0.7, zorder=5)
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

    fut = dft.loc[test_idx].copy()

    label_map = dict(dft[['label', 'label_name']].drop_duplicates().values)
    fut['pred_label_name'] = [label_map.get(int(x), str(int(x))) for x in y_pred]

    fut['month'] = fut['timestamp'].dt.to_period('M').dt.to_timestamp()

    g = fut.groupby(['month', 'label_name']).size().reset_index(name='n')
    g['share'] = g.groupby('month')['n'].transform(lambda x: x / x.sum())

    p = fut.groupby(['month', 'pred_label_name']).size().reset_index(name='n')
    p['share'] = p.groupby('month')['n'].transform(lambda x: x / x.sum())
    p = p.rename(columns={'pred_label_name': 'label_name'})

    labels = ordered_labels(list(g['label_name']) + list(p['label_name']))
    months = sorted(set(fut['month']))
    palette_map = color_map_for_labels(labels)

    plt.figure(figsize=(7.6, 4.4))
    sns.set_style('whitegrid')
    ax = plt.gca()

    for lbl in labels:
        g_lbl = g[g['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})
        p_lbl = p[p['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})
        color = _color_for(lbl, palette_map)
        ax.plot(months, g_lbl['share'].values, '-', color=color, linewidth=3.0, alpha=0.3, zorder=1,
                label=f'{lbl} • ground truth')
        ax.plot(months, p_lbl['share'].values, '-', color=color, linewidth=1.4, alpha=0.95, zorder=3,
                label=f'{lbl} • predicted')

    ax.set_ylabel('Proportion', fontsize=9)
    ax.set_xlabel('Month (future only)', fontsize=9)
    ax.set_title(title or 'Future reconstruction: ground truth vs logistic regression (temporal split)', fontsize=11)
    ax.tick_params(axis='both', labelsize=8)
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
    """
    dft = df.copy()
    if not isinstance(dft['timestamp'].iloc[0], pd.Timestamp):
        dft['timestamp'] = pd.to_datetime(dft['timestamp'])

    fut = dft.loc[test_idx].copy()

    label_map = dict(dft[['label', 'label_name']].drop_duplicates().values)
    fut['pred_label_name'] = [label_map.get(int(x), str(int(x))) for x in y_pred]

    fut['month'] = fut['timestamp'].dt.to_period('M').dt.to_timestamp()

    g = fut.groupby(['month', 'label_name']).size().reset_index(name='n')
    if len(g) == 0:
        raise ValueError('No ground-truth observations in the provided test period (test_idx).')
    g['share'] = g.groupby('month')['n'].transform(lambda x: x / x.sum())

    p = fut.groupby(['month', 'pred_label_name']).size().reset_index(name='n')
    p['share'] = p.groupby('month')['n'].transform(lambda x: x / x.sum())
    p = p.rename(columns={'pred_label_name': 'label_name'})

    labels = ordered_labels(list(g['label_name']) + list(p['label_name']))
    months = sorted(set(fut['month']))
    palette_map = color_map_for_labels(labels)

    n_classes = len(labels)
    if ncols is None or ncols <= 0:
        ncols = n_classes
    nrows = math.ceil(n_classes / ncols)

    fig_w = max(6.0, 2.8 * ncols)
    fig_h = max(2.8, 2.6 * nrows)

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=sharey,
                             figsize=(fig_w, fig_h))

    if isinstance(axes, plt.Axes):
        axes_list = [axes]
    else:
        axes_list = [ax for ax in axes.ravel()]

    for i, lbl in enumerate(labels):
        ax = axes_list[i]
        color = _color_for(lbl, palette_map)

        g_lbl = g[g['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})
        p_lbl = p[p['label_name'] == lbl].set_index('month').reindex(months).fillna({'share': 0})

        ax.plot(months, g_lbl['share'].values, '-', color=color, linewidth=3.0, alpha=0.3, zorder=1,
                label='ground truth')
        ax.plot(months, p_lbl['share'].values, '-', color=color, linewidth=1.4, alpha=0.95, zorder=3,
                label='predicted')

        ax.set_title(lbl, fontsize=9)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if i == 0:
            ax.legend(frameon=True, fontsize=8, loc='upper left')

        if (i // ncols) < (nrows - 1):
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='both', labelsize=8)

    for j in range(len(axes_list)):
        if j >= n_classes:
            axes_list[j].axis('off')

    fig.suptitle(title or 'Future reconstruction by class: ground truth vs prediction', fontsize=11)
    fig.supxlabel('Month (future only)', fontsize=9)
    fig.supylabel('Proportion', fontsize=9)

    for i in range((nrows - 1) * ncols, nrows * ncols):
        if i < len(axes_list):
            for label in axes_list[i].get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)

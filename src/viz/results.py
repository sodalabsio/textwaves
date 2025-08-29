import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def add_value_labels(ax, fmt="{:.3f}"):
    for p in ax.patches:
        height = p.get_height()
        if pd.isna(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )

def plot_methods_grid(csv_path: str, save_path: str | None = None):
    df = pd.read_csv(csv_path)

    # Ensure consistent ordering across subplots
    methods = list(df["Method"].unique())
    df["Method"] = pd.Categorical(df["Method"], categories=methods, ordered=True)

    metrics = [
        ("ARI", "Adjusted Rand Index (higher is better)"),
        ("NMI", "Normalized Mutual Information (higher is better)"),
        ("Silhouette", "Silhouette Score (higher is better)"),
        ("LR_F1_weighted", "Logistic Regression F1 (weighted, higher is better)"),
    ]

    sns.set(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(methods))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        sns.barplot(
            data=df,
            x="Method",
            y=metric,
            hue="Method",  # Assign x variable to hue
            ax=ax,
            palette=palette,
            edgecolor="black",
            legend=False,  # Disable legend since hue is same as x
        )
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)  # these metrics are in [0, 1] here; adjust if needed
        add_value_labels(ax)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Method Comparison Across Metrics", fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 2x2 grid comparing methods across metrics.")
    parser.add_argument("--csv", default="model_results.csv", help="Path to model_results.csv")
    parser.add_argument("--out", default=None, help="Optional path to save the figure (e.g., method_comparison.png)")
    args = parser.parse_args()

    plot_methods_grid(args.csv, args.out)

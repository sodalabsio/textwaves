import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def _short_model(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    # Drop trailing date suffix like -2025-04-14
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    return name


def plot_metrics(summary_path: str, out_path: str):
    """Legacy plot: compare LR vs LLM F1 from the classic metrics_summary_*.csv."""
    if not os.path.exists(summary_path):
        print(f" !! No summary file found at {summary_path}")
        return

    df = pd.read_csv(summary_path)

    # Melt into long format for plotting
    records = []
    for _, row in df.iterrows():
        if not pd.isna(row.get("LR_F1_weighted", float("nan"))):
            records.append({"Method": row["Method"], "Type": "LR", "F1": row["LR_F1_weighted"]})
        if not pd.isna(row.get("LLM_F1_weighted", float("nan"))):
            records.append({"Method": row["Method"], "Type": "LLM", "F1": row["LLM_F1_weighted"]})

    if not records:
        print(" !! No F1 scores found in summary file.")
        return

    plot_df = pd.DataFrame(records)

    plt.figure(figsize=(8, 5))
    for t, sub in plot_df.groupby("Type"):
        plt.bar(sub["Method"], sub["F1"], label=t, alpha=0.7)

    plt.ylabel("F1 Score (weighted)")
    plt.title(f"Comparison of LR vs LLM F1 scores\n{os.path.basename(summary_path)}")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f" --> Saved plot to {out_path}")


def plot_weighted_f1_barchart(metrics_csv: str, out_path: str, sort: str = "desc"):
    """
    New plot for the consolidated metrics file that mixes LR (with train frac) and
    LLM direct prompting (with style + model).

    - Shows one bar per configuration, colored by family (LR method or LLM style).
    - Sorts by F1 (desc by default) to make comparisons easy.
    """
    if not os.path.exists(metrics_csv):
        print(f" !! No metrics file found at {metrics_csv}")
        return

    df = pd.read_csv(metrics_csv)

    rows = []

    # Colors: color-blind friendly-ish palette
    colors_lr = {
        "TFIDF": "#4C78A8",
        "GLOVE": "#72B7B2",
        "SBERT": "#54A24B",
        "OPENAI": "#E45756",
    }
    colors_llm = {
        "extended": "#F58518",
        "zero": "#B279A2",
    }

    # LR entries
    for _, r in df.iterrows():
        f1_lr = r.get("LR_F1_weighted")
        if pd.notna(f1_lr):
            method = str(r.get("Method", ""))
            base = method.replace("-lr", "").upper()
            frac = r.get("TrainFrac")
            frac_s = "" if pd.isna(frac) else f"train={int(round(float(frac)*100))}%"
            label = f"{base} LR\n{frac_s}".strip()
            family = base  # TFIDF/GLOVE/SBERT/OPENAI
            color = colors_lr.get(family, "#666666")
            rows.append({
                "label": label,
                "f1": float(f1_lr),
                "family": f"LR-{family}",
                "color": color,
            })

    # LLM entries
    for _, r in df.iterrows():
        f1_llm = r.get("LLM_F1_weighted")
        if pd.notna(f1_llm):
            method = str(r.get("Method", ""))
            style = method.split("-", 1)[1] if "-" in method else method
            model = _short_model(str(r.get("LLM_model_name", "")))
            style_title = style.title()
            label = f"LLM {style_title}\n{model}".strip()
            color = colors_llm.get(style, "#9C755F")
            rows.append({
                "label": label,
                "f1": float(f1_llm),
                "family": f"LLM-{style}",
                "color": color,
            })

    if not rows:
        print(" !! No weighted F1 scores found in the provided metrics file.")
        return

    plot_df = pd.DataFrame(rows)

    # Sort
    if sort == "desc":
        plot_df = plot_df.sort_values("f1", ascending=False)
    elif sort == "asc":
        plot_df = plot_df.sort_values("f1", ascending=True)

    # Figure sizing based on number of bars
    n = len(plot_df)
    width = max(10, 0.75 * n)
    height = 6

    fig, ax = plt.subplots(figsize=(width, height))

    bars = ax.bar(range(n), plot_df["f1"].values, color=plot_df["color"].values, edgecolor="#333333", linewidth=0.7)

    # X ticks and labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(plot_df["label"].tolist(), rotation=30, ha="right")

    # Y formatting
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate values above bars
    for b, v in zip(bars, plot_df["f1"].values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Legend: unique families in order of appearance
    legend_items = []
    seen = set()
    for fam, col in zip(plot_df["family"].tolist(), plot_df["color"].tolist()):
        if fam not in seen:
            seen.add(fam)
            legend_items.append((fam, col))

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in legend_items]
    labels = [f for f, _ in legend_items]
    ax.legend(handles, labels, title="Family", frameon=False, ncols=min(len(labels), 3))

    title = f"Weighted F1 comparison across LR (by train frac) and LLM prompting\n{os.path.basename(metrics_csv)}"
    ax.set_title(title)

    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f" --> Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize metrics as bar charts")
    parser.add_argument("--data-source", choices=["lexicon", "llm"], default="lexicon",
                        help="Specify the data source, 'lexicon' by default, or 'llm'.")
    parser.add_argument("--legacy", type=bool, default=False,
                        help="Run legacy formulation with some overplotting if needs be.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for the figure. If not set, a sensible default is chosen.")
    parser.add_argument("--sort", choices=["asc", "desc"], default="desc",
                        help="Sorting order by F1 for the consolidated plot.")
    args = parser.parse_args()

    data_source = args.data_source or "lexicon"
    summary_path = os.path.join("outputs", "tables", f"metrics_summary_{data_source}.csv")
    out_path = args.out or os.path.join("outputs", "figures", f"f1_comparison_{data_source}.png")

    if args.legacy:
        plot_metrics(summary_path, out_path)
    else:
        plot_weighted_f1_barchart(summary_path, out_path, sort=args.sort)
        return

if __name__ == "__main__":
    main()

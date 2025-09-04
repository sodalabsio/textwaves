from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd

from ..viz.scatter import scatter_2d
from ..viz.timeseries import plot_monthly_label_proportions, plot_future_reconstruction_by_class


LEX_CSV = os.path.join("synthetic_data", "synthetic.csv")
LLM_CSV = os.path.join("synthetic_data", "synthetic_llm.csv")


def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cache file at {path}")
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d}
    except ValueError as e:
        # Backwards-compat: earlier caches may contain object arrays (e.g., strings)
        # Retry with allow_pickle=True and coerce to safe dtypes when possible.
        if "allow_pickle=False" in str(e):
            with np.load(path, allow_pickle=True) as d:
                out = {k: d[k] for k in d}
            # Coerce common object arrays to unicode strings to avoid downstream issues
            for k, v in list(out.items()):
                if isinstance(v, np.ndarray) and v.dtype == object:
                    try:
                        out[k] = v.astype('U')
                    except Exception:
                        pass
            return out
        raise


def _load_dataset_df(data_source: str, csv_override: str | None) -> pd.DataFrame:
    """Load a dataset DF with at least ['label_name','timestamp'] for timeseries plotting.
    For lexicon/llm, reads the standard synthetic CSVs, unless csv_override is provided.
    For byod, requires csv_override to point to a CSV already normalized by our loaders.
    """
    if csv_override:
        path = csv_override
    else:
        if data_source == "lexicon":
            path = LEX_CSV
        elif data_source == "llm":
            path = LLM_CSV
        else:
            raise SystemExit("For --data-source byod, please supply --csv <path-to-your.csv> containing label_name and timestamp columns.")
    if not os.path.exists(path):
        raise SystemExit(f"Dataset CSV not found at {path}. If you haven't generated data yet, run: python -m src.cli.run_pipeline --regen-data --data-source {data_source}")
    df = pd.read_csv(path)
    if 'timestamp' not in df.columns:
        raise SystemExit("CSV must contain a 'timestamp' column")
    if 'label_name' not in df.columns:
        raise SystemExit("CSV must contain a 'label_name' column")
    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Plot scatter/timeseries/future from cached outputs or from dataset (timeseries)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--what", choices=["scatter", "timeseries", "future"], required=True,
                        help="Which plot to create.")
    parser.add_argument("--data-source", choices=["lexicon", "llm", "byod"], required=True)
    parser.add_argument("--method", choices=["tfidf", "glove", "sbert", "openai"], default=None,
                        help="Required for scatter/future when using default cache; optional for timeseries.")
    parser.add_argument("--cache", default=None, help="Path to a specific cache .npz. If not provided, uses last_{data_source}_{method}.npz for scatter/future.")
    parser.add_argument("--csv", default=None, help="Dataset CSV to use for timeseries (if omitted, lexicon/llm defaults are used; required for byod).")
    parser.add_argument("--cutoff-ts", default=None, help="Optional cutoff timestamp string for the train/test split marker in timeseries (e.g., 2023-06-01).")
    parser.add_argument("--out", default=None, help="Path to write the figure. If not provided, a sensible default is used.")
    args = parser.parse_args()

    # TIMESERIES: can run without cache/method by reading the dataset CSV
    if args.what == "timeseries":
        df = _load_dataset_df(args.data_source, args.csv)
        cutoff = pd.to_datetime(args.cutoff_ts) if args.cutoff_ts else None
        out = args.out or os.path.join("outputs", "figures", f"timeseries_{args.data_source}.png")
        plot_monthly_label_proportions(df.assign(text=df.get('text', '')), out, cutoff_timestamp=cutoff)
        print(f" --> Wrote {out}")
        return

    # SCATTER / FUTURE: use cached arrays
    if args.cache is None:
        if not args.method:
            raise SystemExit("--method is required for scatter/future when --cache is not provided.")
        cache_path = os.path.join("outputs", "cache", f"last_{args.data_source}_{args.method}.npz")
    else:
        cache_path = args.cache

    data = _load_npz(cache_path)

    # Reconstruct a minimal DataFrame for plotting convenience
    label_names = [str(x) for x in data.get("label_names", [])]
    timestamps = pd.to_datetime(data.get("timestamps")) if "timestamps" in data else None
    df = pd.DataFrame({
        "label_name": label_names,
        "timestamp": timestamps,
    })

    if args.what == "scatter":
        coords = data.get("coords2d")
        clabels = data.get("clabels")
        if coords is None or clabels is None or df.empty:
            raise SystemExit("Cache missing coords2d/clabels/labels needed for scatter.")
        out = args.out or os.path.join("outputs", "figures", f"scatter_{args.method}_{args.data_source}.png")
        title = f"{args.method.upper()} • {args.data_source}"
        scatter_2d(coords, df, clabels, title, out)
        print(f" --> Wrote {out}")
        return

    if args.what == "future":
        y_pred = data.get("y_pred")
        test_idx = data.get("test_idx")
        if y_pred is None or test_idx is None or df.empty or df["timestamp"].isna().all():
            raise SystemExit("Cache missing y_pred/test_idx/timestamps for future reconstruction.")
        # Rebuild an index for alignment
        df = df.copy()
        labels = data.get("labels")
        if labels is None:
            labels = np.zeros(len(df), dtype=int)
        else:
            labels = np.asarray(labels).astype(int, copy=False)
        df["label"] = labels
        # Use an integer RangeIndex to match cached indices
        df.index = np.arange(len(df))
        out = args.out or os.path.join("outputs", "figures", f"future_{args.method}_{args.data_source}.png")
        title = f"Future reconstruction • {args.method.upper()} • {args.data_source}"
        plot_future_reconstruction_by_class(df, pd.Index(test_idx), y_pred, out, title=title)
        print(f" --> Wrote {out}")
        return


if __name__ == "__main__":
    main()

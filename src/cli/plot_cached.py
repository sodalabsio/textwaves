from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd

from ..viz.scatter import scatter_2d, scatter_true_only_2d
from ..viz.timeseries import plot_monthly_label_proportions, plot_future_reconstruction_by_class
from ..data.load import load_byod


LEX_CSV = os.path.join('synthetic_data', 'synthetic.csv')
LLM_CSV = os.path.join('synthetic_data', 'synthetic_llm.csv')


def _load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No cache file at {path}')
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d}
    except ValueError as e:
        if 'allow_pickle=False' in str(e):
            with np.load(path, allow_pickle=True) as d:
                out = {k: d[k] for k in d}
            for k, v in list(out.items()):
                if isinstance(v, np.ndarray) and v.dtype == object:
                    try:
                        out[k] = v.astype('U')
                    except Exception:
                        pass
            return out
        raise


def _load_dataset_df(data_source: str, csv_override: str | None,
                     byod_text_col: str = 'text', byod_label_col: str | None = None,
                     byod_timestamp_col: str | None = None) -> pd.DataFrame:
    """Load a dataset DF with at least ['label_name', 'timestamp'] for timeseries plotting.

    - For lexicon/llm, reads the standard synthetic CSVs unless `csv_override` is provided.
    - For byod, `csv_override` should point to the raw user CSV and is normalised via `load_byod`.
    """
    if data_source == 'byod':
        if not csv_override:
            raise SystemExit('For --data-source byod, please supply --csv <path-to-your.csv>. The loader will normalise it using the BYOD column flags.')
        df = load_byod(
            path=csv_override,
            text_col=byod_text_col,
            label_col=byod_label_col,
            timestamp_col=byod_timestamp_col,
        )
        return df

    if csv_override:
        path = csv_override
    else:
        if data_source == 'lexicon':
            path = LEX_CSV
        elif data_source == 'llm':
            path = LLM_CSV
        else:
            raise SystemExit(f'Unsupported data_source: {data_source}')

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
        description='Plot scatter/timeseries/future from cached outputs or from dataset (timeseries)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--what', choices=['scatter', 'scatter-true-only', 'timeseries', 'future'], required=True,
                        help='Which plot to create.')
    parser.add_argument('--data-source', choices=['lexicon', 'llm', 'byod'], required=True)
    parser.add_argument('--method', choices=['tfidf', 'glove', 'sbert', 'openai'], default=None,
                        help='Required for scatter/future when using default cache; optional for timeseries.')
    parser.add_argument('--cache', default=None, help='Path to a specific cache .npz. If not provided, uses last_{data_source}_{method}.npz for scatter/future.')
    parser.add_argument('--csv', default=None, help='Dataset CSV to use for timeseries. For byod, this should be the raw BYOD CSV.')
    parser.add_argument('--cutoff-ts', default=None, help='Optional cutoff timestamp string for the train/test split marker in timeseries (e.g., 2023-06-01).')
    parser.add_argument('--out', default=None, help='Path to write the figure. If not provided, a sensible default is used.')

    # BYOD options for timeseries-from-CSV use.
    parser.add_argument('--byod-text-col', type=str, default='text')
    parser.add_argument('--byod-label-col', type=str, default=None)
    parser.add_argument('--byod-timestamp-col', type=str, default=None)

    args = parser.parse_args()

    if args.what == 'timeseries':
        df = _load_dataset_df(
            args.data_source,
            args.csv,
            byod_text_col=args.byod_text_col,
            byod_label_col=args.byod_label_col,
            byod_timestamp_col=args.byod_timestamp_col,
        )
        cutoff = pd.to_datetime(args.cutoff_ts) if args.cutoff_ts else None
        out = args.out or os.path.join('outputs', 'figures', f'timeseries_{args.data_source}.png')
        plot_monthly_label_proportions(df.assign(text=df.get('text', '')), out, cutoff_timestamp=cutoff)
        print(f' --> Wrote {out}')
        return

    if args.cache is None:
        if not args.method:
            raise SystemExit('--method is required for scatter/future when --cache is not provided.')
        cache_path = os.path.join('outputs', 'cache', f'last_{args.data_source}_{args.method}.npz')
    else:
        cache_path = args.cache

    data = _load_npz(cache_path)

    label_names = [str(x) for x in data.get('label_names', [])]
    timestamps = pd.to_datetime(data.get('timestamps')) if 'timestamps' in data else None
    df = pd.DataFrame({
        'label_name': label_names,
        'timestamp': timestamps,
    })

    if args.what == 'scatter':
        coords = data.get('coords2d')
        clabels = data.get('clabels')
        if coords is None or clabels is None or df.empty:
            raise SystemExit('Cache missing coords2d/clabels/labels needed for scatter.')
        out = args.out or os.path.join('outputs', 'figures', f'scatter_{args.method}_{args.data_source}.png')
        title = f'{args.method.upper()} • {args.data_source}'
        scatter_2d(coords, df, clabels, title, out)
        print(f' --> Wrote {out}')
        return

    if args.what == 'scatter-true-only':
        coords = data.get('coords2d')
        if coords is None or df.empty:
            raise SystemExit('Cache missing coords2d/label_names needed for true-only scatter.')
        out = args.out or os.path.join('outputs', 'figures', f'scatter_true_only_{args.method}_{args.data_source}.png')
        title = f'{args.method.upper()} • {args.data_source} • true labels only'
        scatter_true_only_2d(coords, df, title, out)
        print(f' --> Wrote {out}')
        return

    if args.what == 'future':
        y_pred = data.get('y_pred')
        test_idx = data.get('test_idx')
        if y_pred is None or test_idx is None or df.empty or df['timestamp'].isna().all():
            raise SystemExit('Cache missing y_pred/test_idx/timestamps for future reconstruction.')
        df = df.copy()
        labels = data.get('labels')
        if labels is None:
            labels = np.zeros(len(df), dtype=int)
        else:
            labels = np.asarray(labels).astype(int, copy=False)
        df['label'] = labels
        df.index = np.arange(len(df))
        out = args.out or os.path.join('outputs', 'figures', f'future_{args.method}_{args.data_source}.png')
        title = f'Future reconstruction • {args.method.upper()} • {args.data_source}'
        plot_future_reconstruction_by_class(df, pd.Index(test_idx), y_pred, out, title=title)
        print(f' --> Wrote {out}')
        return


if __name__ == '__main__':
    main()

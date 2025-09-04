from __future__ import annotations
# Centralize env and warnings before heavy imports
from .. import env as _env  # noqa: F401

import os
import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ..config import load_config
from ..utils import ensure_dirs, seed_everything, save_npz, texts_signature
from ..data.load import load_or_generate, load_or_generate_llm, temporal_split, load_byod
from ..vectorize.tfidf import TfidfRepresenter
from ..vectorize.glove import GloveAverager
from ..vectorize.openai_embed import embed as embed_llm
from ..cluster.kmeans import kmeans as kmeans_fit
from ..reduce.umap_ import to_2d as umap_to_2d
from ..reduce.pca import to_2d as pca_to_2d
from ..eval.metrics import cluster_metrics, best_mapping, supervised_scores
from ..viz.scatter import scatter_2d
from ..viz.timeseries import plot_monthly_label_proportions, plot_future_reconstruction_by_class

METHODS = {'tfidf', 'glove', 'openai', 'sbert'}
REDUCERS = {'umap', 'pca'}
CLUSTERERS = {'kmeans'}
DATA_SOURCES = {'lexicon', 'llm', 'byod'}


def _vectorize(method: str, texts: list[str], cfg) -> np.ndarray:
    if method == 'tfidf':
        vcfg = cfg['vectorizers']['tfidf']
        rep = TfidfRepresenter(min_df=vcfg['min_df'], max_df=vcfg['max_df'],
                               ngram_range=tuple(vcfg['ngram_range']))
        X = rep.fit_transform(texts)
        return X.toarray().astype('float32')  # small dataset; ok to densify
    elif method == 'glove':
        model = cfg['vectorizers']['glove']['model']
        rep = GloveAverager(model=model)
        return rep.transform(texts)
    elif method in {'openai', 'sbert'}:
        ocfg = cfg['vectorizers']['openai']
        scfg = cfg['vectorizers']['sbert']
        return embed_llm(
            texts,
            method=method,
            openai_model=ocfg['model'],
            sbert_model=scfg['model'],
            batch_size=ocfg.get('batch_size', 128),
        )
    else:
        raise ValueError('Unknown method')


def _reduce(reduce: str, X: np.ndarray, cfg) -> np.ndarray:
    rcfg = cfg['reduce']
    if reduce == 'umap':
        return umap_to_2d(
            X,
            n_components=rcfg.get('n_components', 2),
            n_neighbors=rcfg.get('n_neighbors', 15),
            seed=cfg['random_seed'],
        )
    return pca_to_2d(X, n_components=rcfg.get('n_components', 2), seed=cfg['random_seed'])


def _cluster(clusterer: str, X: np.ndarray, cfg) -> np.ndarray:
    if clusterer == 'kmeans':
        k = cfg['cluster']['k']
        labels, _ = kmeans_fit(X, k=k, seed=cfg['random_seed'])
        return labels
    raise ValueError('Unknown clusterer')


def synthetic_data_exists(data_source: str) -> bool:
    """Check if synthetic CSVs already exist for the given data source."""
    if data_source == "llm":
        path = "synthetic_data/synthetic_llm.csv"
    else:
        path = "synthetic_data/synthetic.csv"
    return os.path.exists(path)


def main():
    class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description='Run one representation pipeline end-to-end or regenerate data',
        formatter_class=_HelpFormatter,
    )

    parser.add_argument('--method', choices=sorted(METHODS), required=False)
    parser.add_argument('--clusterer', choices=sorted(CLUSTERERS), default=None)
    parser.add_argument('--reduce', choices=sorted(REDUCERS), default=None)
    parser.add_argument('--regen-data', action='store_true')
    parser.add_argument('--train-frac', type=float, default=0.3)
    parser.add_argument('--data-source', choices=sorted(DATA_SOURCES), default='lexicon')
    parser.add_argument('--llm-model', default='gpt-4.1-nano-2025-04-14')

    # BYOD options
    parser.add_argument('--byod-path', type=str, default=None, help='Path to your CSV when --data-source byod')
    parser.add_argument('--byod-text-col', type=str, default='text')
    parser.add_argument('--byod-label-col', type=str, default=None)
    parser.add_argument('--byod-timestamp-col', type=str, default=None)

    # Optional plotting (no recompute required later thanks to cache)
    parser.add_argument('--plot-scatter', action='store_true', help='Save 2D scatter (true color vs cluster marker)')
    parser.add_argument('--plot-timeseries', action='store_true', help='Save monthly label proportions (marks train/test split)')
    parser.add_argument('--plot-future', action='store_true', help='Save future reconstruction overlay by class')

    args = parser.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise SystemExit('--train-frac must be in (0, 1)')

    print(' --> Setting up ...')
    load_dotenv()
    cfg = load_config()
    ensure_dirs()
    seed_everything(cfg['random_seed'])

    if args.clusterer is None:
        args.clusterer = cfg['cluster'].get('algorithm', 'kmeans')
    if args.reduce is None:
        args.reduce = cfg['reduce'].get('method', 'umap')

    # Decide whether to regenerate
    need_regen = args.regen_data or (args.data_source in {"lexicon", "llm"} and not synthetic_data_exists(args.data_source))

    # Standalone regeneration path
    if need_regen and not args.method:
        print(' --> Standalone regeneration path ...')
        if args.data_source == 'llm':
            _ = load_or_generate_llm(
                n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
                add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
                seed=cfg['random_seed'], regen=True, path='synthetic_data/synthetic_llm.csv', openai_model=args.llm_model,
            )
        elif args.data_source == 'lexicon':
            _ = load_or_generate(
                n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
                add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
                seed=cfg['random_seed'], regen=True, path='synthetic_data/synthetic.csv',
            )
        else:
            raise SystemExit('--regen-data is not applicable for byod')
        return

    if not args.method:
        parser.error('When not using --regen-data standalone, you must supply --method.')

    # Load data
    if args.data_source == 'llm':
        df = load_or_generate_llm(
            n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
            add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
            seed=cfg['random_seed'], regen=need_regen, path='synthetic_data/synthetic_llm.csv', openai_model=args.llm_model,
        )
    elif args.data_source == 'lexicon':
        df = load_or_generate(
            n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
            add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
            seed=cfg['random_seed'], regen=need_regen, path='synthetic_data/synthetic.csv',
        )
    else:
        if not args.byod_path:
            parser.error('For --data-source byod you must also provide --byod-path <your.csv>')
        df = load_byod(
            path=args.byod_path,
            text_col=args.byod_text_col,
            label_col=args.byod_label_col,
            timestamp_col=args.byod_timestamp_col,
        )

    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(' --> Vectorizing -> Clustering -> Reducing ...')
    texts = df['text'].tolist()
    X = _vectorize(args.method, texts, cfg)
    clabels = _cluster(args.clusterer, X, cfg)
    coords2d = _reduce(args.reduce, X, cfg)

    # Metrics
    m = cluster_metrics(df['label'].values, clabels, X)
    mapping, cm = best_mapping(df['label'].values, clabels)

    # Train/test split and LR
    print(' --> Training Logistic Regression model ...')
    train_idx, test_idx = temporal_split(df, train_frac=args.train_frac)
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, n_jobs=10))
    clf.fit(X[train_idx], df['label'].values[train_idx])
    y_pred = clf.predict(X[test_idx])

    y_true = df['label'].values[test_idx].astype(str)
    y_pred = y_pred.astype(str)

    s = supervised_scores(y_true, y_pred)

    # Unified metrics row
    print(' --> Saving metrics table ...')
    metrics_path = os.path.join('outputs', 'tables', f'metrics_summary_{args.data_source}.csv')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    row = {
        "Method": f"{args.method}-lr",
        "Data": args.data_source,
        "TrainFrac": args.train_frac,
        "ARI": m["ARI"],
        "NMI": m["NMI"],
        "Silhouette": m["Silhouette"],
        "LR_Accuracy": s["LR_Accuracy"],
        "LR_F1_weighted": s["LR_F1_weighted"],
        "LLM_Accuracy": np.nan,
        "LLM_F1_weighted": np.nan,
    }

    if os.path.exists(metrics_path):
        existing = pd.read_csv(metrics_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        header = ["Method", "Data", "TrainFrac", "ARI", "NMI", "Silhouette",
                  "LR_Accuracy", "LR_F1_weighted", "LLM_Accuracy", "LLM_F1_weighted"]
        updated = pd.DataFrame([row], columns=header)

    updated.to_csv(metrics_path, index=False)
    print(f" --> Saved integrated metrics to {metrics_path}")

    # Cache run products for later plotting without recompute
    print(' --> Caching run products for later plotting ...')
    tag = f"{args.data_source}:{args.method}:{args.reduce}:{args.clusterer}:train={args.train_frac}"
    sig = texts_signature(texts, tag)
    cache_dir = os.path.join('outputs', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'run_{sig}.npz')
    # Coerce to non-object dtypes for pickle-free loading later
    label_names = df['label_name'].astype(str).to_numpy(dtype='U')
    timestamps = df['timestamp'].astype('datetime64[ns]').to_numpy()
    y_pred_u = np.asarray(y_pred, dtype='U')
    save_npz(
        cache_path,
        coords2d=coords2d.astype('float32', copy=False),
        clabels=np.asarray(clabels).astype(int, copy=False),
        labels=np.asarray(df['label'].values).astype(int, copy=False),
        label_names=label_names,
        timestamps=timestamps,
        test_idx=np.asarray(test_idx, dtype=int),
        y_pred=y_pred_u,
        cutoff_ts=df.loc[test_idx, 'timestamp'].min().to_datetime64(),
    )
    # User-friendly handle for latest run by method+data
    last_path = os.path.join(cache_dir, f'last_{args.data_source}_{args.method}.npz')
    try:
        import shutil
        shutil.copyfile(cache_path, last_path)
    except Exception:
        pass

    # Optional plotting
    if args.plot_scatter:
        out = os.path.join('outputs', 'figures', f'scatter_{args.method}_{args.data_source}.png')
        title = f"{args.method.upper()} • {args.data_source}"
        scatter_2d(coords2d, df, clabels, title, out)
        print(f" --> Wrote {out}")

    if args.plot_timeseries:
        out = os.path.join('outputs', 'figures', f'timeseries_{args.method}_{args.data_source}.png')
        cutoff = df.loc[test_idx, 'timestamp'].min()
        plot_monthly_label_proportions(df, out, cutoff_timestamp=cutoff)
        print(f" --> Wrote {out}")

    if args.plot_future:
        out = os.path.join('outputs', 'figures', f'future_{args.method}_{args.data_source}.png')
        title = f"Future reconstruction • {args.method.upper()} • {args.data_source}"
        plot_future_reconstruction_by_class(df, test_idx, y_pred, out, title=title)
        print(f" --> Wrote {out}")

    print(f"Data: {args.data_source}  Method: {args.method}\n"
          f"ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} Silhouette={m['Silhouette']:.3f}\n"
          f"LogReg Acc={s['LR_Accuracy']:.3f} F1_w={s['LR_F1_weighted']:.3f}")


if __name__ == '__main__':
    main()

from __future__ import annotations
import os
import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from ..config import load_config
from ..utils import ensure_dirs, seed_everything, save_table_csv, save_npz
from ..data.load import load_or_generate, load_or_generate_llm, temporal_split
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
DATA_SOURCES = {'lexicon', 'llm'}


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
        # Honor YAML batch_size for OpenAI embedding calls
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
    # Respect YAML n_components for PCA as well
    return pca_to_2d(X, n_components=rcfg.get('n_components', 2), seed=cfg['random_seed'])


def _cluster(clusterer: str, X: np.ndarray, cfg) -> np.ndarray:
    if clusterer == 'kmeans':
        k = cfg['cluster']['k']
        labels, _ = kmeans_fit(X, k=k, seed=cfg['random_seed'])
        return labels
    raise ValueError('Unknown clusterer')


def main():
    # Combine useful formatters: show defaults and allow nicely formatted epilog/examples
    class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    epilog = (
        "Choices explained:\n"
        "\n"
        "  Methods (--method):\n"
        "    tfidf  — TF–IDF bag-of-words vectors\n"
        "    glove  — Average of pretrained GloVe word vectors\n"
        "    openai — OpenAI embeddings via API\n"
        "    sbert  — Sentence-BERT embeddings\n"
        "\n"
        "  Reducers (--reduce):\n"
        "    umap  — Nonlinear UMAP projection to 2D\n"
        "    pca   — Linear PCA projection to 2D\n"
        "\n"
        "  Clusterers (--clusterer):\n"
        "    kmeans — K-means clustering on representation space\n"
        "\n"
        "  Data sources (--data-source):\n"
        "    lexicon — Rule/lexicon-based synthetic dataset\n"
        "    llm     — LLM-generated synthetic dataset (uses --llm-model)\n"
        "\n"
        "Examples:\n"
        "  Regenerate data only (no pipeline run):\n"
        "    python -m package.pipeline_main --regen-data\n"
        "\n"
        "  Run full pipeline with TF–IDF on lexicon data:\n"
        "    python -m package.pipeline_main --method tfidf\n"
        "\n"
        "  Run OpenAI embeddings on LLM data with PCA and KMeans:\n"
        "    python -m package.pipeline_main --method openai --reduce pca --clusterer kmeans --data-source llm\n"
    )

    parser = argparse.ArgumentParser(
        description='Run one representation pipeline end-to-end or regenerate data',
        formatter_class=_HelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        '--method',
        choices=sorted(METHODS),
        required=False,
        metavar='METHOD',
        help='Representation method to run. See choices explained below.',
    )
    parser.add_argument(
        '--clusterer',
        choices=sorted(CLUSTERERS),
        default=None,  # defer to YAML when not provided
        metavar='CLUSTERER',
        help='Clustering algorithm to apply to the representation (defaults to YAML config).',
    )
    parser.add_argument(
        '--reduce',
        choices=sorted(REDUCERS),
        default=None,  # defer to YAML when not provided
        metavar='REDUCER',
        help='Dimensionality reduction to 2D for visualization (defaults to YAML config).',
    )
    parser.add_argument(
        '--regen-data',
        action='store_true',
        help='Regenerate the synthetic dataset. If given without --method, the program regenerates and exits.',
    )
    parser.add_argument(
        '--train-frac',
        type=float,
        default=0.3,
        help='Fraction of earliest timestamps used for training in temporal split (0 < f < 1).',
    )
    parser.add_argument(
        '--data-source',
        choices=sorted(DATA_SOURCES),
        default='lexicon',
        metavar='SOURCE',
        help='Which generator to use for the dataset (see choices below).',
    )
    parser.add_argument(
        '--llm-model',
        default='gpt-4.1-nano-2025-04-14',
        metavar='MODEL',
        help='Model name to use when --data-source llm (ignored otherwise).',
    )

    args = parser.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise SystemExit('--train-frac must be in (0, 1)')

    print(' --> Setting up ...')
    load_dotenv()
    cfg = load_config()
    ensure_dirs()
    seed_everything(cfg['random_seed'])

    # If user did not supply clusterer/reducer, honor YAML values
    if args.clusterer is None:
        args.clusterer = cfg['cluster'].get('algorithm', 'kmeans')
    if args.reduce is None:
        args.reduce = cfg['reduce'].get('method', 'umap')

    # Standalone regeneration mode: allow `--regen-data` without a method
    if args.regen_data and not args.method:
        print(' --> Standalone regeneration path ...')
        if args.data_source == 'llm':
            print('   --> LLM generation initiating ...')
            df = load_or_generate_llm(
                n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
                add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
                seed=cfg['random_seed'], regen=True, path='data/synthetic_llm.csv', openai_model=args.llm_model,
            )
            if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Regenerated {len(df)} rows to data/synthetic_llm.csv with time range {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        else:
            print('   --> Lexicon generation initiating ...')
            df = load_or_generate(
                n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
                add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
                seed=cfg['random_seed'], regen=True, path='data/synthetic.csv',
            )
            if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Regenerated {len(df)} rows to data/synthetic.csv with time range {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        return

    # Normal pipeline requires a method
    if not args.method:
        parser.error('When not using --regen-data standalone, you must supply --method.')

    if args.data_source == 'llm':
        df = load_or_generate_llm(
            n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
            add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
            seed=cfg['random_seed'], regen=args.regen_data, path='data/synthetic_llm.csv', openai_model=args.llm_model,
        )
    else:
        df = load_or_generate(
            n_per_class=cfg['data']['n_per_class'], start=cfg['data']['start'], end=cfg['data']['end'],
            add_label_noise=cfg['data']['add_label_noise'], noise_rate=cfg['data']['noise_rate'],
            seed=cfg['random_seed'], regen=args.regen_data,
        )

    # Ensure timestamp dtype before temporal operations
    if not isinstance(df['timestamp'].iloc[0], pd.Timestamp):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(' --> Vectorizing -> Clustering -> Reducing (dimensionality) ...')
    texts = df['text'].tolist()
    X = _vectorize(args.method, texts, cfg)
    clabels = _cluster(args.clusterer, X, cfg)
    coords2d = _reduce(args.reduce, X, cfg)

    # Metrics
    m = cluster_metrics(df['label'].values, clabels, X)
    mapping, cm = best_mapping(df['label'].values, clabels)

    # Logistic regression baseline (train/test) — temporal split (configurable)
    print(' --> Training Logistic Regression model ...')
    train_idx, test_idx = temporal_split(df, train_frac=args.train_frac)
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, n_jobs=10))
    clf.fit(X[train_idx], df['label'].values[train_idx])
    y_pred = clf.predict(X[test_idx])
    s = supervised_scores(df['label'].values[test_idx], y_pred)

    # Save metrics table (suffix by data source)
    print(' --> Saving metrics table ...')
    suffix = args.data_source
    metrics_path = os.path.join('outputs', 'tables', f'metrics_{args.method}_{suffix}.csv')
    header = ['Method', 'Data', 'ARI', 'NMI', 'Silhouette', 'LR_Accuracy', 'LR_F1_weighted']
    row = [args.method, args.data_source, m['ARI'], m['NMI'], m['Silhouette'], s['LR_Accuracy'], s['LR_F1_weighted']]
    save_table_csv(metrics_path, [row], header)

    # Save scatter
    print(' --> Building scatter plot and saving ...')
    fig_path = os.path.join('outputs', 'figures', f'scatter_{args.method}_{args.data_source}.png')
    scatter_2d(coords2d, df, clabels, title=f'{args.method.upper()} — 2D view ({args.data_source})', path=fig_path)

    # Cache 2D coords
    print(' --> Caching coordinates ...')
    cache_path = os.path.join('outputs', 'cache', f'coords2d_{args.method}_{args.data_source}.npz')
    save_npz(cache_path, coords=coords2d, clabels=clabels)

    # Timeseries
    print(' --> Running time-series analysis and plotting ...')
    # Ground-truth monthly proportions over the whole range with cutoff marker
    cutoff_ts = None
    if len(test_idx) > 0:
        cutoff_ts = df.loc[test_idx, 'timestamp'].min()
    ts_path = os.path.join('outputs', 'figures', f'timeseries_{args.data_source}.png')
    plot_monthly_label_proportions(df, ts_path, cutoff_timestamp=cutoff_ts)

    # Future reconstruction overlay (temporal test period only)
    ts2_path = os.path.join('outputs', 'figures', f'timeseries_future_lr_{args.method}_{args.data_source}_cutoff{args.train_frac}.png')
    title = f"Future reconstruction ({args.method.upper()}, {args.data_source}) — ground truth vs LR"
    plot_future_reconstruction_by_class(
        df,
        test_idx=test_idx,
        y_pred=y_pred,
        path=ts2_path,
        title=title)

    # Console summary
    print(f"Data: {args.data_source}  Method: {args.method}\nARI={m['ARI']:.3f} NMI={m['NMI']:.3f} Silhouette={m['Silhouette']:.3f}\n"
          f"LogReg Acc={s['LR_Accuracy']:.3f} F1_w={s['LR_F1_weighted']:.3f}\nSaved: {fig_path} and {metrics_path}")


if __name__ == '__main__':
    main()

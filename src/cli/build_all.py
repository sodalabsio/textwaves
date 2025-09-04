from __future__ import annotations
# Centralized environment setup (threads, OpenMP warning filters)
from .. import env as _env  # noqa: F401

import os
import subprocess
import sys
import argparse
import shutil
import glob

METHODS = ['tfidf', 'glove', 'sbert', 'openai']

def synthetic_data_exists(data_source: str) -> bool:
    """Check if synthetic CSVs already exist for the given data source."""
    if data_source == "llm":
        pattern = "synthetic_data/synthetic_llm.csv"
    else:
        pattern = "synthetic_data/synthetic.csv"
    return os.path.exists(pattern)

def run_pipeline(method: str, data_source: str, train_frac: float, regen: bool = False):
    cmd = [
        sys.executable, '-m', 'src.cli.run_pipeline',
        '--method', method,
        '--clusterer', 'kmeans',
        '--reduce', 'umap',
        '--data-source', data_source,
        '--train-frac', str(train_frac),
    ]
    if regen:
        cmd.append('--regen-data')
    subprocess.check_call(cmd)

def run_llm_label(data_source: str, prompt_style: str = "zero", regen_labels: bool = False):
    cmd = [
        sys.executable, '-m', 'src.cli.run_llm_label',
        '--data-source', data_source,
        '--prompt-style', prompt_style,
    ]
    if regen_labels:
        cmd.append('--regen-labels')
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(
        description='Run all methods and aggregate metrics into unified summary.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-source',
        choices=['lexicon', 'llm'],
        default='lexicon',
        help='Which dataset source to use for all methods.'
    )
    parser.add_argument(
        '--train-frac',
        type=float,
        default=0.3,
        help='Fraction of earliest timestamps used for training in temporal split (0 < f < 1).'
    )
    parser.add_argument(
        '--include-llm',
        action='store_true',
        help='Also run LLM labelling and include its metrics in the summary.'
    )
    parser.add_argument(
        '--llm-prompt-style',
        choices=['zero', 'extended'],
        default='zero',
        help='Prompt style to use for LLM labelling if included.'
    )
    parser.add_argument(
        '--regen-labels',
        action='store_true',
        help='Force re-run of LLM labelling even if cached labels exist.'
    )
    parser.add_argument(
        '--regen-data',
        action='store_true',
        help='Force regeneration of synthetic data even if CSVs already exist.'
    )
    args = parser.parse_args()

    # Decide whether to regenerate data
    need_regen = args.regen_data or not synthetic_data_exists(args.data_source)

    for i, m in enumerate(METHODS):
        # Only regenerate on the first run if needed
        run_pipeline(m, data_source=args.data_source, train_frac=args.train_frac, regen=(i == 0 and need_regen))

    # Optionally run LLM labelling
    if args.include_llm:
        run_llm_label(args.data_source, prompt_style=args.llm_prompt_style, regen_labels=args.regen_labels)

    # Unified metrics file is already written by run_pipeline.py and run_llm_label.py
    metrics_path = os.path.join('outputs', 'tables', f'metrics_summary_{args.data_source}.csv')

    if os.path.exists(metrics_path):
        print(f'Summary written to {metrics_path}')
    else:
        print(f'Error: expected {metrics_path} not found.')

if __name__ == '__main__':
    main()

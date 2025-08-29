from __future__ import annotations
import os
import subprocess
import sys
import csv
import argparse

METHODS = ['tfidf', 'glove', 'sbert', 'openai']


def run(method: str, data_source: str, regen: bool = False):
    cmd = [
        sys.executable, '-m', 'src.cli.run_pipeline',
        '--method', method,
        '--clusterer', 'kmeans',
        '--reduce', 'umap',
        '--data-source', data_source,
    ]
    if regen:
        cmd.append('--regen-data')
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Run all methods and aggregate metrics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-source',
        choices=['lexicon', 'llm'],
        default='lexicon',
        help='Which dataset source to use for all methods.'
    )
    args = parser.parse_args()

    # First run regenerates data to ensure consistency
    for i, m in enumerate(METHODS):
        run(m, data_source=args.data_source, regen=(i == 0))

    # Aggregate metrics into one summary table for the chosen data source
    rows = []
    header = None
    for m in METHODS:
        p = os.path.join('outputs', 'tables', f'metrics_{m}_{args.data_source}.csv')
        with open(p, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            rows.append(row)

    out_path = os.path.join('outputs', 'tables', f'summary_{args.data_source}.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header if header else ['Method','Data','ARI','NMI','Silhouette','LR_Accuracy','LR_F1_weighted'])
        writer.writerows(rows)
    print(f'Summary written to {out_path}')


if __name__ == '__main__':
    main()

import os
import hashlib
from pathlib import Path
from typing import Iterable
import numpy as np
import random


def ensure_dirs():
    for d in ['data', 'outputs/figures', 'outputs/tables', 'outputs/cache']:
        os.makedirs(d, exist_ok=True)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def texts_signature(texts: Iterable[str], tag: str) -> str:
    h = hashlib.sha1()
    h.update(tag.encode('utf-8'))
    for t in texts:
        h.update(b'\x1f')
        h.update(t.encode('utf-8'))
    return h.hexdigest()[:16]


def save_npz(path: str, **arrays):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path: str):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data}


def save_table_csv(path: str, rows: list, header: list):
    import csv
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

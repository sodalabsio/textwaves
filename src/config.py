import os
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_CONFIG: Dict[str, Any] = {
    'random_seed': 42,
    'data': {
        'n_per_class': 100,
        'start': '2022-01-01',
        'end': '2024-12-31',
        'add_label_noise': False,
        'noise_rate': 0.05,
    },
    'vectorizers': {
        'tfidf': {'min_df': 2, 'max_df': 0.95, 'ngram_range': [1, 2]},
        'glove': {'model': 'glove-wiki-gigaword-50'},
        'openai': {'model': 'text-embedding-3-small', 'batch_size': 128},
        'sbert': {'model': 'all-MiniLM-L6-v2'},
    },
    'cluster': {'algorithm': 'kmeans', 'k': 3},
    'reduce': {'method': 'umap', 'n_components': 2, 'n_neighbors': 15},
}


def load_config(path: str = 'configs/config.yaml') -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if yaml is None:
        return cfg
    if os.path.exists(path):
        with open(path, 'r') as f:
            user = yaml.safe_load(f) or {}
        cfg = _deep_update(cfg, user)
    return cfg


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base

import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from .generate_synthetic import generate as generate_lex, save_csv as save_csv_lex
from .generate_synthetic_llm import generate_llm, save_csv as save_csv_llm


def load_or_generate(n_per_class: int, start: str, end: str, add_label_noise: bool,
                     noise_rate: float, seed: int, regen: bool = False,
                     path: str = 'synthetic_data/synthetic.csv') -> pd.DataFrame:
    if regen or not os.path.exists(path):
        df = generate_lex(n_per_class=n_per_class, start=start, end=end,
                          add_label_noise=add_label_noise, noise_rate=noise_rate, seed=seed)
        save_csv_lex(df, path)
        return df
    return pd.read_csv(path, parse_dates=['timestamp'])


def load_or_generate_llm(n_per_class: int, start: str, end: str, add_label_noise: bool,
                          noise_rate: float, seed: int, regen: bool = False,
                          path: str = 'synthetic_data/synthetic_llm.csv', openai_model: str = 'gpt-4.1-nano-2025-04-14') -> pd.DataFrame:
    if regen or not os.path.exists(path):
        df = generate_llm(n_per_class=n_per_class, start=start, end=end,
                          add_label_noise=add_label_noise, noise_rate=noise_rate, seed=seed,
                          openai_model=openai_model)
        save_csv_llm(df, path)
        return df
    return pd.read_csv(path, parse_dates=['timestamp'])


def stratified_split(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42) -> Tuple[pd.Index, pd.Index]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    y = df['label'].values
    (train_idx, test_idx), = sss.split(df.index.values, y)
    return df.index[train_idx], df.index[test_idx]


def temporal_split(df: pd.DataFrame, train_frac: float = 0.7) -> Tuple[pd.Index, pd.Index]:
    """Split indices by time: first `train_frac` of rows (sorted by timestamp) for train,
    the remainder for test. Returns original index labels for easy selection.
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column for temporal_split")
    d = df.copy()
    if not isinstance(d['timestamp'].iloc[0], pd.Timestamp):
        d['timestamp'] = pd.to_datetime(d['timestamp'])
    # Stable sort so original order breaks ties
    d_sorted = d.sort_values('timestamp', kind='mergesort')
    n = len(d_sorted)
    cut = int(round(n * float(train_frac)))
    cut = max(1, min(n - 1, cut))  # ensure both splits non-empty
    train_orig_idx = d_sorted.index[:cut]
    test_orig_idx = d_sorted.index[cut:]
    return pd.Index(train_orig_idx), pd.Index(test_orig_idx)

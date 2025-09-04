from __future__ import annotations
import os
from typing import Optional, Dict
import pandas as pd


REQUIRED_TEXT_COLS = ["text"]


def _coerce_timestamp(s: pd.Series) -> pd.Series:
    try:
        ts = pd.to_datetime(s, errors="coerce")
        # Fill missing with an ordered index to preserve temporal split behavior
        if ts.isna().any():
            idx_ts = pd.to_datetime(pd.Series(range(len(s))), unit="D", origin="2020-01-01")
            ts = ts.fillna(idx_ts)
        return ts
    except Exception:
        # Fallback: synthetic monotonically increasing timestamps
        return pd.to_datetime(pd.Series(range(len(s))), unit="D", origin="2020-01-01")


def load_byod_csv(
    path: str,
    text_col: str = "text",
    label_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    label_rename: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Load a user's CSV and normalize columns to the repo's schema.

    Returns a DataFrame with columns: text, label_name, label, timestamp.
    If labels are missing, assigns a single class 'unlabeled' (label=0).
    If timestamp is missing/unparseable, creates a synthetic, ordered timestamp.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"BYOD path not found: {path}")

    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Expected a text column '{text_col}' in {path}. Found: {list(df.columns)}")

    # Basic cleanup
    df = df.copy()
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    # Timestamps
    if timestamp_col and timestamp_col in df.columns:
        df["timestamp"] = _coerce_timestamp(df[timestamp_col])
    else:
        df["timestamp"] = _coerce_timestamp(pd.Series([None] * len(df)))

    # Labels
    if label_col and label_col in df.columns:
        lbl = df[label_col].astype(str).str.strip()
        if label_rename:
            lbl = lbl.map(lambda x: label_rename.get(x, x))
        # Stable integer coding by sorted unique names
        cats = sorted(lbl.dropna().unique().tolist())
        code_map = {name: i for i, name in enumerate(cats)}
        df["label_name"] = lbl
        df["label"] = df["label_name"].map(code_map).fillna(-1).astype(int)
    else:
        # Unlabeled data; keep pipeline runnable (clustering still works)
        df["label_name"] = "unlabeled"
        df["label"] = 0

    # Minimum columns for downstream compatibility
    return df[["text", "label_name", "timestamp", "label"]]
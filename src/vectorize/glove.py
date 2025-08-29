from __future__ import annotations
import re
import numpy as np
from typing import List

try:
    import gensim.downloader as api
except Exception as e:  # pragma: no cover
    api = None

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


class GloveAverager:
    def __init__(self, model: str = 'glove-wiki-gigaword-50'):
        self.model_name = model
        self.kv = None

    def _ensure_model(self):
        if self.kv is None:
            if api is None:
                raise RuntimeError('gensim.downloader unavailable; install gensim')
            self.kv = api.load(self.model_name)

    def _embed_text(self, text: str) -> np.ndarray:
        self._ensure_model()
        assert self.kv is not None
        tokens = [t.lower() for t in TOKEN_RE.findall(text)]
        vecs: List[np.ndarray] = []
        for t in tokens:
            if t in self.kv:
                vecs.append(self.kv[t])
        if not vecs:
            return np.zeros(self.kv.vector_size, dtype='float32')
        arr = np.vstack(vecs).astype('float32')
        return arr.mean(axis=0)

    def transform(self, texts: List[str]) -> np.ndarray:
        mats = [self._embed_text(t) for t in texts]
        return np.vstack(mats).astype('float32')

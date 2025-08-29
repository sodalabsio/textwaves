from __future__ import annotations
import os
from typing import List
import numpy as np
from ..utils import texts_signature, save_npz, load_npz


def _try_openai_embed(texts: List[str], model: str, batch_size: int = 128) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI()
    out: List[np.ndarray] = []
    bs = max(1, int(batch_size))
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        resp = client.embeddings.create(model=model, input=batch)
        embs = [np.array(d.embedding, dtype='float32') for d in resp.data]
        out.append(np.vstack(embs))
    X = np.vstack(out)
    # Normalize to unit length (cosine geometry)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / norms).astype('float32')


def _sbert_embed(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    mdl = SentenceTransformer(model_name)
    X = mdl.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(X, dtype='float32')


def embed(
    texts: List[str],
    method: str = 'openai',
    openai_model: str = 'text-embedding-3-small',
    sbert_model: str = 'all-MiniLM-L6-v2',
    cache_dir: str = 'outputs/cache',
    batch_size: int = 128,
) -> np.ndarray:
    if method not in {'openai', 'sbert'}:
        raise ValueError('method must be openai or sbert')

    tag = f'{method}:{openai_model if method=="openai" else sbert_model}:bs={batch_size}'
    sig = texts_signature(texts, tag)
    cache_path = os.path.join(cache_dir, f'emb_{sig}.npz')

    if os.path.exists(cache_path):
        return load_npz(cache_path)['X']

    X: np.ndarray
    if method == 'openai' and os.getenv('OPENAI_API_KEY'):
        try:
            X = _try_openai_embed(texts, openai_model, batch_size=batch_size)
        except Exception as e:  # graceful fallback
            print(f'[openai] Error {e}; falling back to SBERT {sbert_model}')
            X = _sbert_embed(texts, sbert_model)
    else:
        if method == 'openai':
            print('[openai] No OPENAI_API_KEY found; using SBERT fallback')
        X = _sbert_embed(texts, sbert_model)

    save_npz(cache_path, X=X)
    return X

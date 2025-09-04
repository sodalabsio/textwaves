"""
Environment and warnings setup shared by CLI entry points.

Purpose:
- Quiet noisy OpenMP runtime warnings from threadpoolctl when Intel/LLVM OpenMP both load.
- Keep thread counts modest by default to reduce contention and surprises for newcomers.
- Maintain deterministic UMAP by setting n_jobs=1 when random_state is used, and suppress the related warning.

Usage:
Place `import src.utils.env as _env` (or `from ..utils import env as _env`) at the very top of entry-point scripts,
BEFORE importing heavy numerical libraries.

If users want to override these defaults, they can set the corresponding environment variables before running.
"""
from __future__ import annotations
import os
import warnings

# Pragmatic default: allow duplicate OpenMP libraries (esp. on macOS), and keep threads modest
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Silence threadpoolctl warning about dual OpenMP libraries
warnings.filterwarnings(
    "ignore",
    message=r".*Intel OpenMP.*LLVM OpenMP.*",
    category=RuntimeWarning,
    module=r"threadpoolctl",
)

# Silence UMAP message about n_jobs forced to 1 under random_state; we keep determinism by default
warnings.filterwarnings(
    "ignore",
    message=r".*n_jobs value .* overridden to 1 by setting random_state.*",
    category=UserWarning,
    module=r"umap\.umap_",
)

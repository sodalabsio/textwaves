import numpy as np

try:
    import umap
except Exception:
    umap = None


def to_2d(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, seed: int = 42) -> np.ndarray:
    if umap is None:
        from .pca import to_2d as pca_to_2d
        return pca_to_2d(X, n_components=n_components, seed=seed)
    # When random_state is set, UMAP disables parallelism and warns that n_jobs is forced to 1.
    # Set n_jobs=1 explicitly to avoid the warning while keeping determinism.
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=seed,
        n_jobs=1,
    )
    return reducer.fit_transform(X).astype('float32')

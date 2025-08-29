import numpy as np
from sklearn.decomposition import PCA


def to_2d(X: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(X).astype('float32')

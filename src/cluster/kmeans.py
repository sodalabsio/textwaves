from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X: np.ndarray, k: int = 3, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels.astype('int32'), centers.astype('float32')

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix, f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment


def cluster_metrics(true_labels: np.ndarray, cluster_labels: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    sil = silhouette_score(X, cluster_labels, metric='cosine') if len(np.unique(cluster_labels)) > 1 else float('nan')
    return {'ARI': ari, 'NMI': nmi, 'Silhouette': sil}


def best_mapping(true_labels: np.ndarray, cluster_labels: np.ndarray) -> Tuple[Dict[int, int], np.ndarray]:
    cm = confusion_matrix(true_labels, cluster_labels)
    # Hungarian on max => convert to cost by negating
    r_ind, c_ind = linear_sum_assignment(-cm)
    mapping = {c: r for r, c in zip(r_ind, c_ind)}
    return mapping, cm


def supervised_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'LR_Accuracy': accuracy_score(y_true, y_pred),
        'LR_F1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }

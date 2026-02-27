"""
clustering/dbscan.py
---------------------
DBSCAN density-based clustering.

Unlike K-Means, DBSCAN:
    - Does not require pre-specifying the number of clusters
    - Discovers arbitrarily shaped clusters
    - Labels points that belong to no dense region as noise (-1)

The noise label (-1) has direct cybersecurity significance:
    noise points represent sessions too isolated to belong to any
    behavioral group — prime candidates for further anomaly investigation.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def run_dbscan(X: np.ndarray, eps: float, min_samples: int) -> dict:
    """
    Fit DBSCAN on the full feature matrix and return a structured result.

    Parameters:
        X           : Scaled feature matrix (full dataset)
        eps         : Neighbourhood radius in standardised Euclidean space
        min_samples : Minimum points within eps to be classified as a core point

    Returns:
        result (dict) with keys:
            labels      (ndarray) : Cluster label per row (-1 = noise)
            model       (DBSCAN)  : Fitted DBSCAN model
            n_clusters  (int)     : Number of meaningful clusters (excludes noise)
            n_noise     (int)     : Number of noise-labelled points
            noise_pct   (float)   : Noise as a percentage of total records
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)

    n_noise    = int(np.sum(labels == -1))
    n_clusters = len(set(labels) - {-1})
    noise_pct  = round((n_noise / len(labels)) * 100, 4)

    return {
        "labels":     labels,
        "model":      model,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "noise_pct":  noise_pct,
    }

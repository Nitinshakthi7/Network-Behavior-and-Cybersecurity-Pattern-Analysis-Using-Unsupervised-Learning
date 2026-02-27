"""
clustering/kmeans.py
---------------------
K-Means clustering with programmatic best-K selection via Silhouette Score.

Data flow:
    run_kmeans_search(X, k_range, ...) -> result dict
        internally calls run_single_kmeans() per K value
        and silhouette() from evaluation.py

The full dataset is always used for model fitting.
Sampling is applied ONLY during silhouette score calculation.
"""

import numpy as np
from sklearn.cluster import KMeans
from src.clustering.evaluation import silhouette


def run_single_kmeans(X: np.ndarray, k: int, random_state: int) -> tuple:
    """
    Fit K-Means for a single K value on the full dataset.

    Parameters:
        X            : Scaled feature matrix (full dataset)
        k            : Number of clusters
        random_state : Seed for reproducibility

    Returns:
        labels (ndarray) : Cluster assignment for every row in X
        model  (KMeans)  : Fitted model (provides .cluster_centers_)
    """
    model = KMeans(n_clusters=k, init="k-means++", n_init=10,
                   random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model


def run_kmeans_search(X: np.ndarray,
                      k_range: range,
                      sample_size: int,
                      random_state: int) -> dict:
    """
    Search over a range of K values and select the best via Silhouette Score.

    Steps per K:
        1. Fit K-Means on full X
        2. Compute silhouette on a sampled subset
        3. Track the K with the highest score

    Parameters:
        X            : Scaled feature matrix (full dataset)
        k_range      : Iterable of K values to evaluate
        sample_size  : Max rows used for silhouette computation
        random_state : Seed for reproducibility

    Returns:
        result (dict) with keys:
            best_k      (int)      : K selected by highest silhouette score
            labels      (ndarray)  : Cluster labels for best K (full dataset)
            model       (KMeans)   : Fitted model for best K
            centroids   (ndarray)  : Cluster center matrix, shape (K, n_features)
            scores      (dict)     : {k: silhouette_score} for all K tried
    """
    scores     = {}
    best_k     = None
    best_score = -1.0
    best_labels = None
    best_model  = None

    for k in k_range:
        labels, model = run_single_kmeans(X, k, random_state)
        score = silhouette(X, labels, sample_size=sample_size,
                           random_state=random_state)
        scores[k] = score

        if score > best_score:
            best_score  = score
            best_k      = k
            best_labels = labels
            best_model  = model

    return {
        "best_k":    best_k,
        "labels":    best_labels,
        "model":     best_model,
        "centroids": best_model.cluster_centers_,
        "scores":    scores,
    }

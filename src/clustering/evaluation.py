"""
clustering/evaluation.py
-------------------------
Evaluation metrics for clustering results.

Provides:
    silhouette()           — Silhouette Score (sampled for performance)
    cluster_distribution() — Point count per cluster
    cluster_summary_stats()— Mean feature values per cluster (for reporting)
    print_centroids()      — Terminal printout of top distinguishing features
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def silhouette(X: np.ndarray, labels: np.ndarray,
               sample_size: int = 10_000, random_state: int = 42) -> float:
    """
    Compute the Silhouette Score using a sampled subset.

    Noise points (label = -1) are excluded before scoring.
    Returns -1.0 if fewer than 2 valid clusters exist.

    Interpretation:
        -1.0  → worst (points assigned to wrong clusters)
         0.0  → overlapping clusters
        +1.0  → perfectly separated clusters
        >0.25 → acceptable for real-world high-dimensional data
    """
    valid_mask   = labels != -1
    valid_labels = labels[valid_mask]
    valid_X      = X[valid_mask]

    if len(np.unique(valid_labels)) < 2:
        return -1.0

    n = min(sample_size, len(valid_X))
    score = silhouette_score(valid_X, valid_labels,
                             sample_size=n, random_state=random_state)
    return round(float(score), 4)


def cluster_distribution(labels: np.ndarray) -> dict:
    """
    Count points per cluster label.

    Returns:
        dist (dict): {cluster_label: point_count}
                     noise points are included under key -1
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def cluster_summary_stats(X: np.ndarray, labels: np.ndarray,
                          feature_names: list) -> pd.DataFrame:
    """
    Compute mean feature value per cluster.

    Noise points (-1) are included as a separate row for reference.
    Output is a (K x n_features) DataFrame.

    Parameters:
        X             : Scaled feature matrix
        labels        : Cluster assignment array
        feature_names : Column names matching X

    Returns:
        summary (DataFrame): Mean feature values per cluster
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["_cluster"] = labels
    summary = df.groupby("_cluster").mean().reset_index()
    summary = summary.rename(columns={"_cluster": "cluster"})
    return summary


def print_centroids(centroids: np.ndarray, feature_names: list,
                    top_n: int = 5) -> None:
    """
    Print the top-N most distinctive features per cluster centroid.

    Centroid values are in standardised space:
        Positive → feature value above the dataset average for this cluster
        Negative → feature value below the dataset average

    Parameters:
        centroids     : (K, n_features) ndarray from KMeans.cluster_centers_
        feature_names : Feature column names
        top_n         : Number of top/bottom features per cluster to display
    """
    n_clusters = centroids.shape[0]
    print(f"\n{'='*58}")
    print("  Cluster Centroid Analysis")
    print(f"  (standardised values: + above avg, - below avg)")
    print(f"{'='*58}")

    for k in range(n_clusters):
        c = centroids[k]
        high_idx = np.argsort(c)[::-1][:top_n]
        low_idx  = np.argsort(c)[:top_n]

        print(f"\n  Cluster {k}:")
        print(f"    HIGH (above dataset average):")
        for i in high_idx:
            print(f"      {feature_names[i]:<30} {c[i]:+.3f}")
        print(f"    LOW  (below dataset average):")
        for i in low_idx:
            print(f"      {feature_names[i]:<30} {c[i]:+.3f}")

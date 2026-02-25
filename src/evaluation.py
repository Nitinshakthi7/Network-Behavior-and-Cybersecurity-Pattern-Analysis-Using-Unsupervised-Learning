import numpy as np
from sklearn.metrics import silhouette_score


def silhouette(X, labels):
    """
    Compute the Silhouette Score for a given clustering result.

    The silhouette score measures how well each point fits
    its own cluster vs. neighbouring clusters.
    Range: -1 (bad) to +1 (perfect). Above 0.5 is good.

    Parameters:
        X      : Scaled feature matrix
        labels : Cluster assignment array

    Returns:
        score : Silhouette score (float), or -1 if not computable
    """
    unique = np.unique(labels)
    # Remove noise label (-1) for scoring purposes
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_X = X[valid_mask]

    if len(np.unique(valid_labels)) < 2:
        return -1.0

    # Use a sample for large datasets to keep computation fast
    sample_size = min(10000, len(valid_X))
    score = silhouette_score(valid_X, valid_labels, sample_size=sample_size, random_state=42)
    return round(score, 4)


def cluster_distribution(labels, title="Cluster Distribution"):
    """
    Print a simple count of how many points fall in each cluster.

    Parameters:
        labels : Cluster assignment array
        title  : Label for the printout
    """
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n[{title}]")
    for label, count in zip(unique, counts):
        tag = "(noise)" if label == -1 else ""
        print(f"  Cluster {label:>3} {tag:<8}: {count} points")


def print_centroid_analysis(km_model, feature_names: list, top_n: int = 5):
    """
    Analyse K-Means cluster centroids to identify what makes each cluster unique.

    For each cluster:
      - Prints the top N features with the highest centroid value  (above average)
      - Prints the top N features with the lowest centroid value   (below average)

    Centroid values are in standardised (scaled) space:
      - Positive value → feature is above the dataset average for this cluster
      - Negative value → feature is below the dataset average for this cluster

    Parameters:
        km_model     : Fitted KMeans model (has .cluster_centers_)
        feature_names: List of feature column names (same order as X)
        top_n        : Number of top/bottom features to show per cluster
    """
    centers = km_model.cluster_centers_   # shape: (K, n_features)
    n_clusters = centers.shape[0]

    print(f"\n{'='*55}")
    print("  K-Means Cluster Centroid Analysis")
    print(f"{'='*55}")
    print("  (Values are standardised: + = above avg, - = below avg)\n")

    for k in range(n_clusters):
        centroid = centers[k]

        # Sort feature indices by centroid value descending / ascending
        top_high_idx  = np.argsort(centroid)[::-1][:top_n]   # highest features
        top_low_idx   = np.argsort(centroid)[:top_n]          # lowest features

        n_points = "(small cluster)" if (centroid == centers[k]).all() else ""

        print(f"  Cluster {k}:")
        print(f"    Distinctly HIGH features (above average for this cluster):")
        for idx in top_high_idx:
            print(f"      {feature_names[idx]:<25} centroid = {centroid[idx]:+.3f}")

        print(f"    Distinctly LOW features (below average for this cluster):")
        for idx in top_low_idx:
            print(f"      {feature_names[idx]:<25} centroid = {centroid[idx]:+.3f}")
        print()

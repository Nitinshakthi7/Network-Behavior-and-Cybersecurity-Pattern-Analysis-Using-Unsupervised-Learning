import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from src.evaluation import silhouette


# ──────────────────────────────────────────────
#  K-MEANS CLUSTERING
# ──────────────────────────────────────────────

def run_kmeans(X, k):
    """
    Fit K-Means for a given K value.

    Parameters:
        X : Scaled feature matrix
        k : Number of clusters

    Returns:
        labels : Cluster assignment array
        model  : Fitted KMeans object
    """
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X)
    return labels, model


def find_best_k(X, k_range):
    """
    Try multiple K values, compute Silhouette Score for each,
    and return the best K along with all scores.

    Parameters:
        X       : Scaled feature matrix
        k_range : List or range of K values to try

    Returns:
        best_k      : K with highest silhouette score
        best_labels : Cluster labels for best K
        scores      : Dict of {k: silhouette_score}
    """
    scores = {}
    best_k = None
    best_score = -1
    best_labels = None

    print("\n[K-Means] Trying different K values...")
    for k in k_range:
        labels, _ = run_kmeans(X, k)
        score = silhouette(X, labels)
        scores[k] = score
        print(f"  K = {k}  |  Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"\n[K-Means] Best K = {best_k}  |  Silhouette Score = {best_score:.4f}")
    return best_k, best_labels, scores


# ──────────────────────────────────────────────
#  DBSCAN CLUSTERING
# ──────────────────────────────────────────────

def run_dbscan(X, eps, min_samples):
    """
    Fit DBSCAN with given parameters.

    Parameters:
        X           : Scaled feature matrix
        eps         : Maximum distance between two points in a neighbourhood
        min_samples : Minimum points required to form a core point

    Returns:
        labels : Cluster assignment array (-1 = noise)
        model  : Fitted DBSCAN object
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)
    return labels, model


def evaluate_dbscan(X, labels, eps, min_samples):
    """
    Evaluate DBSCAN output and print key statistics.

    Parameters:
        X           : Scaled feature matrix
        labels      : Cluster labels from DBSCAN
        eps         : eps value used
        min_samples : min_samples value used
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})       # exclude noise label
    n_noise = np.sum(labels == -1)
    noise_pct = (n_noise / len(labels)) * 100

    print(f"\n[DBSCAN] eps={eps}, min_samples={min_samples}")
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise} ({noise_pct:.2f}% of data)")

    # Silhouette score only makes sense with 2+ clusters
    if n_clusters >= 2:
        score = silhouette(X, labels)
        print(f"  Silhouette Score : {score:.4f}")
    else:
        print("  Silhouette Score : Not applicable (fewer than 2 clusters)")

    return n_clusters, n_noise

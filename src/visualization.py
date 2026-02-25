import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
SAMPLE_SIZE = 10_000   # rows used for scatter / silhouette plots


# ── 1. Silhouette Score vs K ──────────────────────────────────────────────────

def plot_silhouette_vs_k(km_scores: dict, best_k: int, save_path: str = None):
    """
    Line plot of Silhouette Score for each K value tested in K-Means.

    Parameters:
        km_scores : {k: silhouette_score} dict from find_best_k()
        best_k    : K selected as best
        save_path : Optional file path to save the figure
    """
    k_values = list(km_scores.keys())
    scores   = list(km_scores.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(k_values, scores, marker="o", linewidth=2,
            color="#4C72B0", markerfacecolor="white", markeredgewidth=2, markersize=8)

    # Highlight best K
    best_score = km_scores[best_k]
    ax.scatter([best_k], [best_score], color="#DD8452", s=120, zorder=5,
               label=f"Best K = {best_k}  (score = {best_score:.4f})")

    ax.set_title("K-Means: Silhouette Score vs Number of Clusters (K)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Silhouette Score", fontsize=11)
    ax.set_xticks(k_values)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(scores) + 0.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved: {save_path}")
    plt.show()


# ── 2. K-Means Cluster Size Distribution ─────────────────────────────────────

def plot_kmeans_cluster_sizes(km_labels: np.ndarray, best_k: int, save_path: str = None):
    """
    Bar chart showing how many data points fall in each K-Means cluster.

    Parameters:
        km_labels : Cluster label array from K-Means
        best_k    : Best K (used in title)
        save_path : Optional file path to save the figure
    """
    unique, counts = np.unique(km_labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(unique, counts, color=sns.color_palette("muted", len(unique)),
                  edgecolor="white", linewidth=0.8)

    # Annotate each bar with count
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{count:,}", ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_title(f"K-Means Cluster Size Distribution  (K = {best_k})",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Cluster Label", fontsize=11)
    ax.set_ylabel("Number of Points", fontsize=11)
    ax.set_xticks(unique)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved: {save_path}")
    plt.show()


# ── 3. DBSCAN Summary Bar Chart ───────────────────────────────────────────────

def plot_dbscan_summary(db_labels: np.ndarray, save_path: str = None):
    """
    Bar chart summarising DBSCAN output:
      - Total meaningful clusters found
      - Total noise points flagged

    Does NOT plot each cluster individually.

    Parameters:
        db_labels : Cluster label array from DBSCAN (-1 = noise)
        save_path : Optional file path to save the figure
    """
    n_clusters = len(set(db_labels) - {-1})
    n_noise    = int(np.sum(db_labels == -1))
    n_clustered = len(db_labels) - n_noise

    categories = ["Clusters Found", "Noise Points", "Clustered Points"]
    values     = [n_clusters, n_noise, n_clustered]
    colors     = ["#4C72B0", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("DBSCAN Clustering Summary", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved: {save_path}")
    plt.show()


# ── 4. 2D Scatter Plot — K-Means Clusters ────────────────────────────────────

def plot_kmeans_scatter(X: np.ndarray, km_labels: np.ndarray,
                        feature_names: list, best_k: int,
                        feat_idx_1: int = 0, feat_idx_2: int = 1,
                        save_path: str = None):
    """
    2D scatter plot of K-Means clusters using two selected features.
    Uses a random sample to avoid freezing on large datasets.

    Parameters:
        X            : Scaled feature matrix (numpy array)
        km_labels    : K-Means cluster label array
        feature_names: List of feature column names
        best_k       : Best K (used in title)
        feat_idx_1   : Column index for X-axis feature
        feat_idx_2   : Column index for Y-axis feature
        save_path    : Optional file path to save the figure
    """
    # Sample for performance
    n = len(X)
    sample_idx = np.random.default_rng(42).choice(n, size=min(SAMPLE_SIZE, n), replace=False)

    X_sample      = X[sample_idx]
    labels_sample = km_labels[sample_idx]

    x_vals = X_sample[:, feat_idx_1]
    y_vals = X_sample[:, feat_idx_2]

    x_name = feature_names[feat_idx_1] if feature_names else f"Feature {feat_idx_1}"
    y_name = feature_names[feat_idx_2] if feature_names else f"Feature {feat_idx_2}"

    palette = sns.color_palette("tab10", best_k)
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(sorted(set(labels_sample)))}
    point_colors = [color_map[l] for l in labels_sample]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(x_vals, y_vals, c=point_colors, s=8, alpha=0.5, linewidths=0)

    # Legend patches
    patches = [mpatches.Patch(color=palette[i], label=f"Cluster {i}") for i in range(best_k)]
    ax.legend(handles=patches, title="Cluster", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, title_fontsize=9)

    ax.set_title(f"K-Means Cluster Scatter  (K={best_k}, sample={SAMPLE_SIZE:,} pts)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(x_name, fontsize=11)
    ax.set_ylabel(y_name, fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved: {save_path}")
    plt.show()

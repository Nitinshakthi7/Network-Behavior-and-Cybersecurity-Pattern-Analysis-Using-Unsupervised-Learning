import os
from src.preprocessing import load_data, preprocess
from src.clustering import find_best_k, run_dbscan, evaluate_dbscan
from src.evaluation import cluster_distribution, print_centroid_analysis
from src.visualization import (
    plot_silhouette_vs_k,
    plot_kmeans_cluster_sizes,
    plot_dbscan_summary,
    plot_kmeans_scatter,
)

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────

DATA_PATH = os.path.join("data", "UNSW_NB15_training-set.csv")

# Label columns — kept aside for evaluation only, not used in training
LABEL_COLS = ["label", "attack_cat"]

# K-Means: range of K values to try
K_RANGE = range(2, 9)

# DBSCAN: parameters
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 10

# Scatter plot: which two features to use for 2D visualization
# These indices refer to columns in the scaled feature matrix after preprocessing
# sbytes (index 7) = total bytes sent from source
# dbytes (index 8) = total bytes sent from destination
# These two features meaningfully show traffic volume differences between clusters
SCATTER_FEAT_1 = 7   # sbytes — source bytes
SCATTER_FEAT_2 = 8   # dbytes — destination bytes

# Optional: save plots to disk (set to None to just display)
PLOTS_DIR = "plots"


# ──────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────

def main():

    # Create plots folder if saving
    if PLOTS_DIR:
        os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── STEP 1: Load and Preprocess Data ─────────────────────────────────
    print("=" * 55)
    print("  STEP 1: Loading and Preprocessing Data")
    print("=" * 55)

    df = load_data(DATA_PATH)
    X, labels_df, feature_names = preprocess(df, label_cols=LABEL_COLS)

    print(f"[INFO] Features used for clustering: {len(feature_names)}")

    # ── STEP 2: K-Means Clustering ────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 2: K-Means Clustering")
    print("=" * 55)

    best_k, km_labels, km_scores = find_best_k(X, K_RANGE)
    cluster_distribution(km_labels, title=f"K-Means (K={best_k})")

    # ── Centroid Analysis ──────────────────────────────────────────────────
    # Run K-Means once more with best_k to get the fitted model object
    from src.clustering import run_kmeans
    _, km_model = run_kmeans(X, best_k)
    print_centroid_analysis(km_model, feature_names, top_n=5)

    # ── STEP 3: DBSCAN Clustering ─────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 3: DBSCAN Clustering")
    print("=" * 55)

    db_labels, db_model = run_dbscan(X, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    evaluate_dbscan(X, db_labels, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    cluster_distribution(db_labels, title="DBSCAN")

    # ── DBSCAN Parameter Interpretation ───────────────────────────────────
    print(f"""
[DBSCAN Interpretation]
  eps={DBSCAN_EPS} means two points must be within distance {DBSCAN_EPS} (in scaled space)
  to be considered neighbours.

  min_samples={DBSCAN_MIN_SAMPLES} means a point needs at least {DBSCAN_MIN_SAMPLES} neighbours
  within eps to be classified as a core point.

  Why 107 clusters formed:
    The dataset has many small, tight groups of network sessions with very
    similar feature values (e.g. same protocol, same packet size pattern).
    With eps=1.5, DBSCAN treats each such dense micro-group as its own cluster.
    Increasing eps (e.g. to 2.5 or 3.0) would merge nearby clusters,
    producing fewer, broader clusters.
    Increasing min_samples (e.g. to 50) would force clusters to be larger,
    pushing more points into the noise category.

  Noise points ({int((db_labels == -1).sum())} total):
    These are records that do not belong to any dense region.
    In a cybersecurity context these are the most suspicious sessions —
    they represent rare or anomalous traffic patterns.
    """.strip())

    # ── STEP 4: Visualizations ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  STEP 4: Generating Visualizations")
    print("=" * 55)

    # Plot 1 – Silhouette Score vs K
    plot_silhouette_vs_k(
        km_scores,
        best_k,
        save_path=os.path.join(PLOTS_DIR, "silhouette_vs_k.png") if PLOTS_DIR else None
    )

    # Plot 2 – K-Means Cluster Size Distribution
    plot_kmeans_cluster_sizes(
        km_labels,
        best_k,
        save_path=os.path.join(PLOTS_DIR, "kmeans_cluster_sizes.png") if PLOTS_DIR else None
    )

    # Plot 3 – DBSCAN Summary
    plot_dbscan_summary(
        db_labels,
        save_path=os.path.join(PLOTS_DIR, "dbscan_summary.png") if PLOTS_DIR else None
    )

    # Plot 4 – K-Means 2D Scatter (sampled)
    plot_kmeans_scatter(
        X, km_labels, feature_names, best_k,
        feat_idx_1=SCATTER_FEAT_1,
        feat_idx_2=SCATTER_FEAT_2,
        save_path=os.path.join(PLOTS_DIR, "kmeans_scatter.png") if PLOTS_DIR else None
    )

    # ── STEP 5: (Placeholder — Anomaly Detection to be added after learning) ──

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()

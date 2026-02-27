"""
main.py
-------
Entry point for the Unsupervised Behavioral Network Traffic Analysis Framework.

Pipeline order:
    1. Setup — logging, output directories
    2. Load data
    3. Feature engineering (configurable)
    4. Preprocessing — label separation, encoding, scaling
    5. K-Means clustering — K selection via silhouette search
    6. K-Means evaluation — centroid analysis, cluster distribution
    7. DBSCAN clustering
    8. DBSCAN evaluation
    9. Visualization — 4 plots saved to outputs/plots/
   10. Reporting — text report + CSV saved to outputs/reports/
"""

import config
from src.utils import setup_logger, ensure_dirs, timer
from src.preprocessing import load_data, preprocess
from src.feature_engineering import apply_feature_engineering
from src.clustering.kmeans import run_kmeans_search
from src.clustering.dbscan import run_dbscan
from src.clustering.evaluation import (
    cluster_distribution,
    cluster_summary_stats,
    print_centroids,
)
from src.visualization import (
    plot_silhouette_vs_k,
    plot_kmeans_cluster_sizes,
    plot_dbscan_summary,
    plot_2d_scatter,
)
from src.reporting import generate_text_report, export_cluster_summary_csv

import os


def main():

    # ── Setup ─────────────────────────────────────────────────────────────────
    logger = setup_logger("framework")
    ensure_dirs(config.PLOTS_DIR, config.MODELS_DIR, config.REPORTS_DIR)
    logger.info("Framework initialised.")
    logger.info(f"Output directories: {config.OUTPUTS_DIR}/")

    # ── Step 1: Load Data ─────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1 — Loading Dataset")
    logger.info("=" * 55)

    df = load_data(config.DATA_PATH)
    logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 2 — Feature Engineering")
    logger.info("=" * 55)

    if config.FEATURE_ENGINEERING_ENABLED:
        df = apply_feature_engineering(df)
        logger.info("Engineered features added: outbound_dominance_ratio, "
                    "packet_rate, bytes_per_packet, packet_asymmetry")
    else:
        logger.info("Feature engineering disabled (FEATURE_ENGINEERING_ENABLED=False)")

    # ── Step 3: Preprocessing ─────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 3 — Preprocessing")
    logger.info("=" * 55)

    X, labels_df, feature_names, scaler = preprocess(df, label_cols=config.LABEL_COLS)
    total_records = len(X)
    logger.info(f"Labels separated: {config.LABEL_COLS}")
    logger.info(f"Features after preprocessing: {len(feature_names)}")
    logger.info(f"Feature matrix shape: {X.shape}")

    # ── Step 4: K-Means Clustering ────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 4 — K-Means Clustering")
    logger.info("=" * 55)

    k_range = range(config.KMEANS_K_MIN, config.KMEANS_K_MAX + 1)
    logger.info(f"Testing K values: {list(k_range)}")
    logger.info(f"Silhouette sample size: {config.KMEANS_SILHOUETTE_SAMPLE_SIZE:,}")

    km_result = run_kmeans_search(
        X,
        k_range=k_range,
        sample_size=config.KMEANS_SILHOUETTE_SAMPLE_SIZE,
        random_state=config.RANDOM_STATE,
    )

    best_k = km_result["best_k"]
    logger.info(f"Best K selected: {best_k}  |  "
                f"Silhouette Score: {km_result['scores'][best_k]:.4f}")

    # ── Step 5: K-Means Evaluation ────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 5 — K-Means Evaluation")
    logger.info("=" * 55)

    dist_km = cluster_distribution(km_result["labels"])
    logger.info("Cluster distribution:")
    for label, count in dist_km.items():
        pct = (count / total_records) * 100
        logger.info(f"  Cluster {label:>3}: {count:>7,} sessions ({pct:.2f}%)")

    print_centroids(km_result["centroids"], feature_names, top_n=5)

    km_summary = cluster_summary_stats(X, km_result["labels"], feature_names)

    # ── Step 6: DBSCAN Clustering ─────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 6 — DBSCAN Clustering")
    logger.info("=" * 55)

    logger.info(f"DBSCAN parameters: eps={config.DBSCAN_EPS}, "
                f"min_samples={config.DBSCAN_MIN_SAMPLES}")

    db_result = run_dbscan(X, eps=config.DBSCAN_EPS,
                            min_samples=config.DBSCAN_MIN_SAMPLES)

    logger.info(f"Clusters found : {db_result['n_clusters']}")
    logger.info(f"Noise points   : {db_result['n_noise']:,} "
                f"({db_result['noise_pct']:.2f}% of data)")

    dist_db = cluster_distribution(db_result["labels"])
    logger.info(f"DBSCAN cluster labels: {len(dist_db)} groups (including noise at -1)")

    # ── Step 7: Visualization ─────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 7 — Generating Visualizations")
    logger.info("=" * 55)

    plot_silhouette_vs_k(
        scores=km_result["scores"],
        best_k=best_k,
        save_path=os.path.join(config.PLOTS_DIR, "silhouette_vs_k.png"),
    )
    logger.info("Saved: silhouette_vs_k.png")

    plot_kmeans_cluster_sizes(
        labels=km_result["labels"],
        best_k=best_k,
        total=total_records,
        save_path=os.path.join(config.PLOTS_DIR, "kmeans_cluster_sizes.png"),
    )
    logger.info("Saved: kmeans_cluster_sizes.png")

    plot_dbscan_summary(
        db_result=db_result,
        total=total_records,
        save_path=os.path.join(config.PLOTS_DIR, "dbscan_summary.png"),
    )
    logger.info("Saved: dbscan_summary.png")

    plot_2d_scatter(
        X=X,
        labels=km_result["labels"],
        feature_names=feature_names,
        best_k=best_k,
        feat_x=config.SCATTER_FEAT_X,
        feat_y=config.SCATTER_FEAT_Y,
        sample_size=config.SCATTER_SAMPLE_SIZE,
        save_path=os.path.join(config.PLOTS_DIR, "kmeans_scatter.png"),
    )
    logger.info("Saved: kmeans_scatter.png")

    # ── Step 8: Reporting ─────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 8 — Generating Reports")
    logger.info("=" * 55)

    report_path = os.path.join(config.REPORTS_DIR, "summary.txt")
    generate_text_report(
        km_result=km_result,
        db_result=db_result,
        feature_names=feature_names,
        total_records=total_records,
        rare_threshold=config.RARE_CLUSTER_THRESHOLD,
        save_path=report_path,
    )
    logger.info(f"Saved: {report_path}")

    csv_path = os.path.join(config.REPORTS_DIR, "cluster_summary.csv")
    export_cluster_summary_csv(km_summary, save_path=csv_path)
    logger.info(f"Saved: {csv_path}")

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("Framework pipeline complete.")
    logger.info(f"All outputs written to: {config.OUTPUTS_DIR}/")
    logger.info("=" * 55)

    # ── Future modules (placeholder) ──────────────────────────────────────────
    # When ready, add the following steps here:
    #   from src.anomaly.isolation_forest import run_isolation_forest
    #   from src.association.apriori import run_apriori
    #   from src.recommendation.policy import suggest_rules
    #   from src.search.query import run_query


if __name__ == "__main__":
    main()

"""
main.py
-------
Entry point for the Unsupervised Behavioral Network Traffic Analysis Framework.

Pipeline order:
    1.  Setup              — logging, output directories
    2.  Load data
    3.  Feature engineering (configurable)
    4.  Preprocessing      — label separation, encoding, scaling
    5.  K-Means clustering — K selection via silhouette search
    6.  K-Means evaluation — centroid analysis, cluster distribution
    7.  DBSCAN clustering
    8.  Visualization      — 6 core plots saved to outputs/plots/
    9.  Association Rules  — Apriori on discretised traffic features
    10. Label Evaluation   — cluster purity + ARI vs attack_cat
    11. Isolation Forest   — global anomaly detection + SOC priority queue
    12. Reporting          — text report + CSVs saved to outputs/reports/
"""

import os
import pandas as pd
import config
from src.utils import setup_logger, ensure_dirs
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
    plot_top_rules,
    plot_if_results,
)
from src.reporting import generate_text_report, export_cluster_summary_csv
from src.association.apriori import run_apriori, save_rules_to_csv
from src.evaluation_labels import (
    evaluate_clusters_against_labels,
    print_label_evaluation,
)
from src.anomaly.isolation_forest import (
    run_isolation_forest,
    print_if_results,
)


def main():

    # ── Setup ─────────────────────────────────────────────────────────────────
    logger = setup_logger("framework")
    ensure_dirs(config.PLOTS_DIR, config.MODELS_DIR, config.REPORTS_DIR)
    logger.info("Framework initialised.")

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
    logger.info(f"Testing K = {list(k_range)}")
    logger.info(f"Silhouette sample size: {config.KMEANS_SILHOUETTE_SAMPLE_SIZE:,}")

    km_result = run_kmeans_search(
        X,
        k_range=k_range,
        sample_size=config.KMEANS_SILHOUETTE_SAMPLE_SIZE,
        random_state=config.RANDOM_STATE,
    )

    best_k = km_result["best_k"]
    logger.info(f"Best K = {best_k}  |  "
                f"Silhouette = {km_result['scores'][best_k]:.4f}")

    # ── Step 5: K-Means Evaluation ────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 5 — K-Means Evaluation")
    logger.info("=" * 55)

    dist_km = cluster_distribution(km_result["labels"])
    for label, count in dist_km.items():
        pct = (count / total_records) * 100
        logger.info(f"  Cluster {label:>3}: {count:>7,} sessions ({pct:.2f}%)")

    print_centroids(km_result["centroids"], feature_names, top_n=5)
    km_summary = cluster_summary_stats(X, km_result["labels"], feature_names)

    # ── Step 6: DBSCAN Clustering ─────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 6 — DBSCAN Clustering")
    logger.info("=" * 55)

    logger.info(f"DBSCAN eps={config.DBSCAN_EPS}, "
                f"min_samples={config.DBSCAN_MIN_SAMPLES}")

    db_result = run_dbscan(X, eps=config.DBSCAN_EPS,
                            min_samples=config.DBSCAN_MIN_SAMPLES)

    logger.info(f"Clusters found : {db_result['n_clusters']}")
    logger.info(f"Noise points   : {db_result['n_noise']:,} "
                f"({db_result['noise_pct']:.2f}%)")

    # ── Step 7: Core Visualizations ───────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 7 — Core Visualizations")
    logger.info("=" * 55)

    plot_silhouette_vs_k(
        scores=km_result["scores"], best_k=best_k,
        save_path=os.path.join(config.PLOTS_DIR, "silhouette_vs_k.png"),
    )
    plot_kmeans_cluster_sizes(
        labels=km_result["labels"], best_k=best_k, total=total_records,
        save_path=os.path.join(config.PLOTS_DIR, "kmeans_cluster_sizes.png"),
    )
    plot_dbscan_summary(
        db_result=db_result, total=total_records,
        save_path=os.path.join(config.PLOTS_DIR, "dbscan_summary.png"),
    )
    plot_2d_scatter(
        X=X, labels=km_result["labels"], feature_names=feature_names,
        best_k=best_k, feat_x=config.SCATTER_FEAT_X, feat_y=config.SCATTER_FEAT_Y,
        sample_size=config.SCATTER_SAMPLE_SIZE,
        save_path=os.path.join(config.PLOTS_DIR, "kmeans_scatter.png"),
    )
    logger.info("Core visualizations saved.")

    # ── Step 8: Association Rule Mining ───────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 8 — Association Rule Mining (Apriori)")
    logger.info("=" * 55)
    logger.info(f"Input  : post-feature-engineering, unscaled DataFrame "
                f"({df.shape[0]:,} rows x {df.shape[1]} cols)")
    logger.info(f"min_support    : {config.APRIORI_MIN_SUPPORT}")
    logger.info(f"min_confidence : {config.APRIORI_MIN_CONFIDENCE}")

    # run_apriori receives `df` — the DataFrame AFTER apply_feature_engineering
    # but BEFORE preprocess() scales it. Label columns (label, attack_cat) may
    # still be present but are silently ignored inside _build_transactions.
    apriori_result = run_apriori(
        df,
        min_support=config.APRIORI_MIN_SUPPORT,
        min_confidence=config.APRIORI_MIN_CONFIDENCE,
    )

    n_itemsets = len(apriori_result["frequent_itemsets"])
    n_rules    = len(apriori_result["rules"])
    logger.info(f"Frequent itemsets mined : {n_itemsets:,}")
    logger.info(f"Association rules found : {n_rules:,}")

    # Save all rules to a single CSV
    arm_csv_path = os.path.join(config.REPORTS_DIR, "association_rules.csv")
    save_rules_to_csv(apriori_result["rules"], save_path=arm_csv_path)
    logger.info(f"ARM rules saved : {arm_csv_path}")

    # Visualise top-20 rules by lift
    plot_top_rules(
        rules=apriori_result["rules"],
        top_n=20,
        save_path=os.path.join(config.PLOTS_DIR, "association_rules.png"),
    )
    logger.info("Saved: association_rules.png")

    # ── Step 9: Label Evaluation ──────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 9 — Cluster Evaluation Against Attack Labels")
    logger.info("=" * 55)

    eval_result = evaluate_clusters_against_labels(
        km_labels=km_result["labels"],
        labels_df=labels_df,
    )

    print_label_evaluation(eval_result)
    logger.info(f"Overall Cluster Purity : {eval_result['overall_purity']:.4f}")
    logger.info(f"Adjusted Rand Index    : {eval_result['ari']:.4f}")

    # Save cross-tab CSV
    crosstab_path = os.path.join(config.REPORTS_DIR, "label_crosstab.csv")
    eval_result["crosstab"].to_csv(crosstab_path)
    logger.info(f"Saved: label_crosstab.csv")

    # Save dominant category summary CSV
    dom_path = os.path.join(config.REPORTS_DIR, "dominant_category_per_cluster.csv")
    eval_result["dominant_summary"].to_csv(dom_path, index=False)
    logger.info(f"Saved: dominant_category_per_cluster.csv")

    # ── Step 11: Isolation Forest ────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 11 -- Isolation Forest Anomaly Detection")
    logger.info("=" * 55)
    logger.info(f"contamination={config.ANOMALY_CONTAMINATION}  "
                f"n_estimators={config.ANOMALY_N_ESTIMATORS}")

    # Build DBSCAN noise mask for dual-method agreement
    import numpy as np
    dbscan_noise_mask = db_result["labels"] == -1

    if_result = run_isolation_forest(
        X                  = X,
        km_labels          = km_result["labels"],
        feature_names      = feature_names,
        contamination      = config.ANOMALY_CONTAMINATION,
        n_estimators       = config.ANOMALY_N_ESTIMATORS,
        random_state       = config.RANDOM_STATE,
        dbscan_noise_mask  = dbscan_noise_mask,
    )
    print_if_results(if_result)

    # Save SOC priority queue and cluster breakdown
    if_result["soc_queue"].head(config.ANOMALY_SOC_TOP_N).to_csv(
        os.path.join(config.REPORTS_DIR, "if_soc_queue.csv"), index=False)
    if_result["summary_df"].to_csv(
        os.path.join(config.REPORTS_DIR, "if_cluster_summary.csv"), index=False)
    logger.info("Saved: if_soc_queue.csv, if_cluster_summary.csv")

    # Anomaly plot
    plot_if_results(
        if_result = if_result,
        km_result = km_result,
        save_path = os.path.join(config.PLOTS_DIR, "isolation_forest.png"),
    )
    logger.info(f"Isolation Forest complete: {if_result['n_anomalies']:,} anomalies "
                f"({if_result['anomaly_pct']}%)  "
                f"| Dual-flagged: {if_result['dual_flagged']:,}")

    # ── Step 14: Reporting ──────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 14 -- Generating Reports")
    logger.info("=" * 55)

    generate_text_report(
        km_result      = km_result,
        db_result      = db_result,
        feature_names  = feature_names,
        total_records  = total_records,
        rare_threshold = config.RARE_CLUSTER_THRESHOLD,
        save_path      = os.path.join(config.REPORTS_DIR, "summary.txt"),
    )
    export_cluster_summary_csv(
        km_summary,
        save_path = os.path.join(config.REPORTS_DIR, "cluster_summary.csv"),
    )
    logger.info("All reports saved.")

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("Framework pipeline complete.")
    logger.info(f"All outputs written to: {config.OUTPUTS_DIR}/")
    logger.info("=" * 55)

    # ── Future integration points ─────────────────────────────────────────────
    # Step 15 -- LOF (Local Outlier Factor):
    #   from src.anomaly.lof import run_lof
    #   lof_result = run_lof(X, km_labels, feature_names)


if __name__ == "__main__":
    main()

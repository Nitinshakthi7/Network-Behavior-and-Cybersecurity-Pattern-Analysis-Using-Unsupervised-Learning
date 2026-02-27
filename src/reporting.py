"""
reporting.py
------------
Generates structured text and CSV outputs from clustering results.

Outputs:
    outputs/reports/summary.txt       — Human-readable text summary
    outputs/reports/cluster_summary.csv — Per-cluster mean feature stats

The text report is designed to be submission-ready for academic and
enterprise reporting purposes.
"""

import os
import pandas as pd


def generate_text_report(km_result: dict,
                          db_result: dict,
                          feature_names: list,
                          total_records: int,
                          rare_threshold: float,
                          save_path: str) -> None:
    """
    Write a structured summary report to a text file.

    Covers:
        - K-Means: best K, silhouette scores, cluster sizes, rare cluster flags
        - DBSCAN: parameters, cluster count, noise rate
        - Behavioral interpretation placeholders for future analysis phases

    Parameters:
        km_result       : Result dict from run_kmeans_search()
        db_result       : Result dict from run_dbscan()
        feature_names   : List of feature column names
        total_records   : Total rows in the dataset
        rare_threshold  : Fraction below which a cluster is flagged as rare
        save_path       : File path for the saved report
    """
    km_labels    = km_result["labels"]
    best_k       = km_result["best_k"]
    scores       = km_result["scores"]
    best_score   = scores[best_k]
    centroids    = km_result["centroids"]

    # Cluster sizes
    import numpy as np
    unique, counts = np.unique(km_labels, return_counts=True)
    rare_clusters = [int(u) for u, c in zip(unique, counts)
                     if c / total_records < rare_threshold]

    lines = []
    lines.append("=" * 65)
    lines.append("  BEHAVIORAL NETWORK TRAFFIC ANALYSIS — CLUSTERING REPORT")
    lines.append("=" * 65)
    lines.append("")

    # ── Dataset ──
    lines.append("DATASET SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Total records        : {total_records:,}")
    lines.append(f"  Feature count        : {len(feature_names)}")
    lines.append("")

    # ── K-Means ──
    lines.append("K-MEANS CLUSTERING")
    lines.append("-" * 40)
    lines.append(f"  Best K selected      : {best_k}")
    lines.append(f"  Selection method     : Maximum Silhouette Score")
    lines.append(f"  Best Silhouette Score: {best_score:.4f}")
    lines.append("")
    lines.append("  Silhouette scores per K:")
    for k, s in scores.items():
        marker = " ← selected" if k == best_k else ""
        lines.append(f"    K = {k:>2}  →  {s:.4f}{marker}")
    lines.append("")
    lines.append("  Cluster size distribution:")
    for u, c in zip(unique, counts):
        pct  = (c / total_records) * 100
        flag = "  [RARE — < 1% traffic]" if u in rare_clusters else ""
        lines.append(f"    Cluster {u:>3}  :  {c:>7,} sessions  ({pct:5.2f}%){flag}")
    lines.append("")
    if rare_clusters:
        lines.append(f"  Rare clusters identified: {rare_clusters}")
        lines.append("  → These clusters warrant further investigation as potential")
        lines.append("    attack signatures or edge-case traffic patterns.")
    lines.append("")

    # ── DBSCAN ──
    lines.append("DBSCAN CLUSTERING")
    lines.append("-" * 40)
    lines.append(f"  eps               : {db_result.get('model').eps}")
    lines.append(f"  min_samples       : {db_result.get('model').min_samples}")
    lines.append(f"  Clusters found    : {db_result['n_clusters']}")
    lines.append(f"  Noise points      : {db_result['n_noise']:,}  ({db_result['noise_pct']:.2f}% of data)")
    lines.append("")
    lines.append("  Noise point interpretation:")
    lines.append("    Sessions labelled as noise do not belong to any dense")
    lines.append("    behavioral region. In a security context, these are the")
    lines.append("    primary candidates for anomalous or attack traffic.")
    lines.append("")

    # ── Behavioral Placeholders ──
    lines.append("BEHAVIORAL INTERPRETATION (PLACEHOLDER)")
    lines.append("-" * 40)
    lines.append("  Cluster semantic labeling will be completed after cross-")
    lines.append("  referencing cluster assignments with attack_cat labels.")
    lines.append("")
    lines.append("FUTURE MODULES")
    lines.append("-" * 40)
    lines.append("  [ ] Anomaly Detection       — Isolation Forest / LOF")
    lines.append("  [ ] Association Rule Mining — Apriori / FP-Growth")
    lines.append("  [ ] Recommender System      — Firewall rule suggestion")
    lines.append("  [ ] Keyword / Query Search  — Interactive cluster filtering")
    lines.append("")
    lines.append("=" * 65)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_cluster_summary_csv(summary_df: pd.DataFrame, save_path: str) -> None:
    """
    Export the per-cluster mean feature stats DataFrame as a CSV.

    Parameters:
        summary_df : DataFrame from cluster_summary_stats()
        save_path  : File path for the saved CSV
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary_df.to_csv(save_path, index=False)

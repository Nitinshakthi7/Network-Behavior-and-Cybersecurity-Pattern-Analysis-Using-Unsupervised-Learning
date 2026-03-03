"""
evaluation_labels.py
---------------------
Formal evaluation of clustering results against ground-truth attack labels.

This module answers the question:
    "Do our behavioral segments correspond to real attack categories?"

Labels (attack_cat) were excluded from all model training.
This module uses them ONLY for post-hoc validation — confirming that
unsupervised clustering discovered structure aligned with real attack types.

Metrics computed:
    Cluster Purity        — what fraction of each cluster is its dominant class?
    Overall Purity        — weighted average purity across all clusters
    Adjusted Rand Index   — statistical measure of cluster-label agreement
    Cross-tabulation      — full cluster × attack_cat frequency matrix
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def compute_crosstab(km_labels: np.ndarray,
                     attack_cat: pd.Series) -> pd.DataFrame:
    """
    Build a cross-tabulation of cluster label vs attack category.

    Parameters:
        km_labels  : K-Means cluster label array (length n)
        attack_cat : attack_cat column from labels_df (length n)

    Returns:
        crosstab (DataFrame): rows = clusters, columns = attack categories,
                              values = session counts
    """
    df = pd.DataFrame({
        "cluster":    km_labels,
        "attack_cat": attack_cat.values,
    })
    return pd.crosstab(df["cluster"], df["attack_cat"])


def compute_purity(crosstab: pd.DataFrame) -> tuple:
    """
    Compute per-cluster and overall cluster purity.

    Purity per cluster = (count of dominant attack category) / (cluster size)
    Overall purity     = sum(dominant counts) / total sessions

    Returns:
        per_cluster (Series): purity value per cluster label
        overall     (float) : weighted overall purity
    """
    dominant_counts = crosstab.max(axis=1)
    cluster_totals  = crosstab.sum(axis=1)
    per_cluster     = (dominant_counts / cluster_totals).round(4)
    overall         = round(dominant_counts.sum() / cluster_totals.sum(), 4)
    return per_cluster, overall


def compute_ari(km_labels: np.ndarray, attack_cat: pd.Series) -> float:
    """
    Compute the Adjusted Rand Index between cluster assignments and labels.

    ARI = 1.0  → perfect agreement
    ARI = 0.0  → agreement no better than random
    ARI < 0.0  → worse than random (very unlikely with real structure)

    Parameters:
        km_labels  : K-Means cluster labels
        attack_cat : Ground-truth attack category labels (string or encoded)
    """
    # Encode string categories to integers for ARI computation
    cat_encoded = pd.Categorical(attack_cat).codes
    return round(float(adjusted_rand_score(cat_encoded, km_labels)), 4)


def dominant_category_per_cluster(crosstab: pd.DataFrame) -> pd.DataFrame:
    """
    Return the most common attack category and its percentage for each cluster.

    Returns:
        summary (DataFrame): cluster, dominant_category, sessions_in_category,
                             cluster_total, purity_%
    """
    rows = []
    for cluster in crosstab.index:
        row       = crosstab.loc[cluster]
        dom_cat   = row.idxmax()
        dom_count = row.max()
        total     = row.sum()
        rows.append({
            "cluster":            int(cluster),
            "dominant_category":  dom_cat,
            "sessions_in_cat":    int(dom_count),
            "cluster_total":      int(total),
            "purity_%":           round((dom_count / total) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("cluster")


def evaluate_clusters_against_labels(km_labels: np.ndarray,
                                     labels_df: pd.DataFrame) -> dict:
    """
    Run the full label evaluation pipeline.

    Parameters:
        km_labels  : K-Means cluster label array
        labels_df  : DataFrame containing 'attack_cat' and optionally 'label'

    Returns:
        result (dict):
            crosstab      (DataFrame) : full cross-tabulation
            per_purity    (Series)    : purity per cluster
            overall_purity(float)     : weighted overall purity
            ari           (float)     : Adjusted Rand Index
            dominant_summary(DataFrame): dominant category per cluster
    """
    attack_cat = labels_df["attack_cat"].fillna("Normal")

    crosstab         = compute_crosstab(km_labels, attack_cat)
    per_purity, opurity = compute_purity(crosstab)
    ari              = compute_ari(km_labels, attack_cat)
    dom_summary      = dominant_category_per_cluster(crosstab)

    return {
        "crosstab":        crosstab,
        "per_purity":      per_purity,
        "overall_purity":  opurity,
        "ari":             ari,
        "dominant_summary": dom_summary,
    }


def print_label_evaluation(result: dict) -> None:
    """Print the label evaluation results to the terminal."""
    print(f"\n{'='*60}")
    print("  Cluster Evaluation Against Attack Labels")
    print(f"{'='*60}")
    print(f"  Overall Cluster Purity : {result['overall_purity']:.4f}")
    print(f"  Adjusted Rand Index    : {result['ari']:.4f}")
    print(f"\n  Dominant Attack Category per Cluster:")
    print(f"  {'Cluster':<10} {'Dominant Category':<20} "
          f"{'Count':>8} {'Total':>8} {'Purity':>8}")
    print(f"  {'-'*58}")
    for _, row in result["dominant_summary"].iterrows():
        print(f"  {int(row['cluster']):<10} {str(row['dominant_category']):<20} "
              f"{int(row['sessions_in_cat']):>8,} "
              f"{int(row['cluster_total']):>8,} "
              f"{row['purity_%']:>7.1f}%")

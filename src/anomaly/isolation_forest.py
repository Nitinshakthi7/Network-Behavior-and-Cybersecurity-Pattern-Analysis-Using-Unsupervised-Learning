"""
isolation_forest.py
-------------------
Isolation Forest anomaly detection for behavioral network traffic analysis.

Adapted from teammate Greeshma's implementation (8-step notebook),
integrated into the existing pipeline framework.

Key adaptations vs the original notebook:
    - Uses the pipeline's already-scaled X (no re-scaling)
    - Uses the pipeline's already-separated feature_names (no re-dropping cols)
    - Enriches output with K-Means cluster labels for cross-method comparison
    - Adds cross-validation with DBSCAN noise points
    - Saves SOC priority queue and anomaly summary as CSVs

Original notebook logic preserved:
    - n_estimators=200, contamination=0.04, random_state=42
    - anomaly_label (-1 = anomaly, 1 = normal)
    - anomaly_flag (1 = anomaly, 0 = normal)
    - anomaly_score (decision_function — lower = more anomalous)
    - SOC priority queue sorted by anomaly_score ascending

Business Value:
    Isolation Forest detects global outliers — sessions that are structurally
    different from the majority of traffic across ALL features simultaneously.
    Combined with DBSCAN (density-based) it provides dual-method anomaly
    confidence: sessions flagged by BOTH are highest-priority threats.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ── Core entry point ──────────────────────────────────────────────────────────

def run_isolation_forest(X: np.ndarray,
                          km_labels: np.ndarray,
                          feature_names: list,
                          contamination: float = 0.04,
                          n_estimators: int = 200,
                          random_state: int = 42,
                          dbscan_noise_mask: np.ndarray = None) -> dict:
    """
    Fit Isolation Forest on the scaled feature matrix and return anomaly results.

    Preserves the teammate's 8-step logic, adapted for the pipeline:
        Step 1 — Libraries          (handled at module import)
        Step 2 — Load dataset       (X already loaded + scaled by pipeline)
        Step 3 — Select features    (feature_names already cleaned by pipeline)
        Step 4 — Feature scaling    (X already StandardScaler-scaled)
        Step 5 — Train IF           (this function)
        Step 6 — Generate scores    (decision_function → anomaly_score)
        Step 7 — Check results      (summary dict)
        Step 8 — SOC priority queue (sorted by anomaly_score ascending)

    Parameters:
        X                : Scaled feature matrix — shape (n_samples, n_features)
        km_labels        : K-Means cluster label per session (n_samples,)
        feature_names    : Feature column names matching X
        contamination    : Expected anomaly fraction (default 0.04 = 4%)
        n_estimators     : Number of isolation trees (default 200)
        random_state     : Reproducibility seed
        dbscan_noise_mask: Optional boolean mask — True where DBSCAN flagged
                           noise. Used to compute dual-method agreement.

    Returns:
        dict with keys:
            model         : Fitted IsolationForest object
            anomaly_flag  : int array — 1=anomaly, 0=normal  (n_samples,)
            anomaly_score : float array — decision score, lower=more anomalous
            n_anomalies   : Count of flagged sessions
            anomaly_pct   : Percentage of flagged sessions
            soc_queue     : DataFrame — top anomalies sorted by risk score
            summary_df    : DataFrame — anomalies per K-Means cluster
            dual_flagged  : int — sessions flagged by BOTH IF and DBSCAN
    """
    total = len(X)

    # ── Step 5: Train Isolation Forest ───────────────────────────────────────
    print(f"[INFO] Training Isolation Forest  "
          f"(n_estimators={n_estimators}, contamination={contamination})")

    iso_forest = IsolationForest(
        n_estimators  = n_estimators,
        contamination = contamination,
        random_state  = random_state,
        n_jobs        = -1,
    )
    iso_forest.fit(X)

    # ── Step 6: Generate anomaly scores ──────────────────────────────────────
    raw_pred      = iso_forest.predict(X)          # -1 or +1
    anomaly_flag  = (raw_pred == -1).astype(int)   # 1 = anomaly, 0 = normal
    anomaly_score = iso_forest.decision_function(X)  # lower = more anomalous

    # ── Step 7: Check results ─────────────────────────────────────────────────
    n_anomalies  = int(anomaly_flag.sum())
    anomaly_pct  = round((n_anomalies / total) * 100, 2)

    print(f"[INFO] Total Sessions       : {total:,}")
    print(f"[INFO] Isolated (anomalies) : {n_anomalies:,}")
    print(f"[INFO] Isolation Rate       : {anomaly_pct}%")

    # ── Step 8: SOC priority queue ────────────────────────────────────────────
    soc_df = pd.DataFrame({
        "session_idx":   np.arange(total),
        "km_cluster":    km_labels,
        "anomaly_flag":  anomaly_flag,
        "anomaly_score": anomaly_score,
    })
    for j, fname in enumerate(feature_names):
        soc_df[fname] = X[:, j]

    soc_queue = (soc_df[soc_df["anomaly_flag"] == 1]
                 .sort_values("anomaly_score")  # lowest score = most anomalous
                 .reset_index(drop=True))

    # ── Per-cluster breakdown ─────────────────────────────────────────────────
    cluster_ids  = np.unique(km_labels)
    summary_rows = []
    for cid in cluster_ids:
        mask      = km_labels == cid
        n_total   = int(mask.sum())
        n_flagged = int(anomaly_flag[mask].sum())
        summary_rows.append({
            "cluster":           int(cid),
            "sessions_in_cluster": n_total,
            "if_anomalies":      n_flagged,
            "if_anomaly_rate_%": round(n_flagged / n_total * 100, 2),
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── Dual-method agreement (IF + DBSCAN) ───────────────────────────────────
    dual_flagged = 0
    if dbscan_noise_mask is not None:
        dual_flagged = int((anomaly_flag.astype(bool) & dbscan_noise_mask).sum())
        dual_pct     = round(dual_flagged / max(n_anomalies, 1) * 100, 1)
        print(f"[INFO] Dual-flagged (IF + DBSCAN) : {dual_flagged:,} "
              f"({dual_pct}% of IF anomalies) — HIGHEST PRIORITY")

    return {
        "model":         iso_forest,
        "anomaly_flag":  anomaly_flag,
        "anomaly_score": anomaly_score,
        "n_anomalies":   n_anomalies,
        "anomaly_pct":   anomaly_pct,
        "soc_queue":     soc_queue,
        "summary_df":    summary_df,
        "dual_flagged":  dual_flagged,
    }


def print_if_results(result: dict) -> None:
    """Print a compact terminal summary of Isolation Forest results."""
    print(f"\n{'='*58}")
    print(f"ISOLATION FOREST — Anomaly Detection Results")
    print(f"{'='*58}")
    print(f"  Flagged sessions    : {result['n_anomalies']:,} "
          f"({result['anomaly_pct']}% of traffic)")
    print(f"  Dual-flagged (+ DBSCAN): {result['dual_flagged']:,}  "
          f"← CRITICAL PRIORITY")
    print(f"\n  Top 10 Highest-Risk Sessions (SOC Queue):")
    print(f"  {'Rank':<5} {'Session':<10} {'Cluster':<10} {'Score':<12}")
    print(f"  {'-'*40}")
    for i, row in result["soc_queue"].head(10).iterrows():
        print(f"  {i+1:<5} {int(row['session_idx']):<10,} "
              f"{int(row['km_cluster']):<10} "
              f"{row['anomaly_score']:<12.5f}")
    print()
    print(f"  Per-Cluster Breakdown:")
    print(result["summary_df"].to_string(index=False))
    print()

"""
visualization.py
----------------
All plot generation for the clustering pipeline.

Plots are saved to the path specified in config.PLOTS_DIR.
All plots use a consistent dark-themed professional style.
Sampling is applied for scatter plots to avoid rendering lag on large datasets.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")

_ACCENT  = "#00B4D8"
_YELLOW  = "#FFD166"
_RED     = "#EF476F"
_BG      = "#0D1B2A"
_TEXT    = "#CAEEFF"


def _apply_dark_bg(fig, ax):
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor("#1A2D42")
    ax.tick_params(colors=_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A4060")


def plot_silhouette_vs_k(scores: dict, best_k: int, save_path: str) -> None:
    """
    Line plot of Silhouette Score vs K.

    Business context title: shows how cluster separation quality
    varies with the number of behavioral groups.

    Parameters:
        scores    : {k: silhouette_score} dict
        best_k    : K with the highest score (highlighted)
        save_path : Full file path to save the figure
    """
    k_vals  = list(scores.keys())
    s_vals  = list(scores.values())
    best_sc = scores[best_k]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_bg(fig, ax)

    ax.plot(k_vals, s_vals, marker="o", linewidth=2.2, color=_ACCENT,
            markerfacecolor="white", markeredgewidth=2, markersize=9)
    ax.scatter([best_k], [best_sc], s=140, color=_YELLOW, zorder=5,
               label=f"Best K = {best_k}  (score = {best_sc:.4f})")

    ax.set_title("Silhouette Score vs Number of Clusters (K)\n"
                 "Network Traffic Behavioral Segmentation Quality",
                 fontsize=12, fontweight="bold", pad=10, color=_TEXT)
    ax.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax.set_ylabel("Silhouette Score", fontsize=10)
    ax.set_xticks(k_vals)
    ax.legend(fontsize=9, facecolor=_BG, labelcolor=_TEXT)
    ax.set_ylim(0, max(s_vals) + 0.05)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=_BG)
    plt.show()
    plt.close()


def plot_kmeans_cluster_sizes(labels: np.ndarray, best_k: int,
                              total: int, save_path: str) -> None:
    """
    Bar chart of K-Means cluster sizes with % of total traffic annotated.

    Parameters:
        labels    : K-Means cluster label array
        best_k    : K used (for title)
        total     : Total dataset record count (for % calculation)
        save_path : Full file path to save the figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    pcts = (counts / total) * 100
    colors = sns.color_palette("muted", len(unique))

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_bg(fig, ax)

    bars = ax.bar(unique, counts, color=colors, edgecolor="white", linewidth=0.7)
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.003,
                f"{count:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8.5, color=_TEXT)

    ax.set_title(f"K-Means Cluster Size Distribution  (K = {best_k})\n"
                 "Traffic Volume per Behavioral Segment",
                 fontsize=12, fontweight="bold", pad=10, color=_TEXT)
    ax.set_xlabel("Cluster Label", fontsize=10)
    ax.set_ylabel("Number of Sessions", fontsize=10)
    ax.set_xticks(unique)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=_BG)
    plt.show()
    plt.close()


def plot_dbscan_summary(db_result: dict, total: int, save_path: str) -> None:
    """
    Bar chart summarising DBSCAN output:
        - Number of clusters discovered
        - Noise points (potential anomalies)
        - Clustered points (assigned to a group)

    Parameters:
        db_result : Result dict from run_dbscan()
        total     : Total dataset record count
        save_path : Full file path to save the figure
    """
    n_clusters  = db_result["n_clusters"]
    n_noise     = db_result["n_noise"]
    n_clustered = total - n_noise

    cats   = ["Clusters\nDiscovered", "Noise Points\n(Potential Anomalies)", "Clustered\nSessions"]
    vals   = [n_clusters, n_noise, n_clustered]
    colors = [_ACCENT, _RED, "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_bg(fig, ax)

    bars = ax.bar(cats, vals, color=colors, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f"{val:,}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=_TEXT)

    ax.set_title("DBSCAN Density-Based Clustering Summary\n"
                 "Cluster Discovery and Noise (Anomaly Candidate) Analysis",
                 fontsize=12, fontweight="bold", pad=10, color=_TEXT)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.15)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=_BG)
    plt.show()
    plt.close()


def plot_2d_scatter(X: np.ndarray, labels: np.ndarray,
                    feature_names: list, best_k: int,
                    feat_x: str, feat_y: str,
                    sample_size: int, save_path: str) -> None:
    """
    2D scatter projection of K-Means clusters using two selected features.
    A random sample is used to prevent rendering lag on large datasets.

    Parameters:
        X             : Scaled feature matrix
        labels        : K-Means label array
        feature_names : Column names matching X
        best_k        : Number of clusters (for title)
        feat_x        : Feature name for X-axis
        feat_y        : Feature name for Y-axis
        sample_size   : Max rows to plot
        save_path     : Full file path to save the figure
    """
    if feat_x not in feature_names or feat_y not in feature_names:
        print(f"[WARN] Scatter feature '{feat_x}' or '{feat_y}' not found. Skipping scatter plot.")
        return

    xi = feature_names.index(feat_x)
    yi = feature_names.index(feat_y)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    Xs  = X[idx, xi]
    Ys  = X[idx, yi]
    Ls  = labels[idx]

    palette  = sns.color_palette("tab10", best_k)
    c_map    = {lbl: palette[i % len(palette)] for i, lbl in enumerate(sorted(set(Ls)))}
    pt_colors = [c_map[l] for l in Ls]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_bg(fig, ax)

    ax.scatter(Xs, Ys, c=pt_colors, s=7, alpha=0.45, linewidths=0)

    patches = [mpatches.Patch(color=palette[i], label=f"Cluster {i}")
               for i in range(best_k)]
    ax.legend(handles=patches, title="Cluster", bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=8, title_fontsize=9,
              facecolor=_BG, labelcolor=_TEXT)

    ax.set_title(f"K-Means Cluster Projection: {feat_x} vs {feat_y}\n"
                 f"K={best_k}  |  Sample={sample_size:,} sessions",
                 fontsize=12, fontweight="bold", pad=10, color=_TEXT)
    ax.set_xlabel(f"{feat_x}  (standardised)", fontsize=10)
    ax.set_ylabel(f"{feat_y}  (standardised)", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=_BG)
    plt.show()
    plt.close()

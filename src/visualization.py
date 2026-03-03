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

    palette   = sns.color_palette("tab10", best_k)
    c_map     = {lbl: palette[i % len(palette)] for i, lbl in enumerate(sorted(set(Ls)))}
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


# ── Association Rule Visualization ────────────────────────────────────────────

_TOKEN_LABELS = {
    "sbytes_High":   "High Outbound Bytes",   "sbytes_Medium": "Medium Outbound Bytes",
    "sbytes_Low":    "Low Outbound Bytes",     "dbytes_High":   "High Inbound Bytes",
    "dbytes_Medium": "Medium Inbound Bytes",   "dbytes_Low":    "Low Inbound Bytes",
    "spkts_High":    "High Source Packets",    "spkts_Medium":  "Medium Source Packets",
    "spkts_Low":     "Low Source Packets",     "dpkts_High":    "High Dest Packets",
    "dpkts_Medium":  "Medium Dest Packets",    "dpkts_Low":     "Low Dest Packets",
    "dur_High":      "Long Session Duration",  "dur_Medium":    "Medium Session Duration",
    "dur_Low":       "Short Session Duration", "rate_High":     "High Transfer Rate",
    "rate_Medium":   "Medium Transfer Rate",   "rate_Low":      "Low Transfer Rate",
    "sttl_High":     "High Source TTL",        "sttl_Low":      "Low Source TTL",
    "dttl_High":     "High Dest TTL",          "dttl_Low":      "Low Dest TTL",
    "sload_High":    "High Source Load",       "sload_Low":     "Low Source Load",
    "dload_High":    "High Dest Load",         "dload_Low":     "Low Dest Load",
    "outbound_dominance_ratio_High":   "High Outbound Dominance",
    "outbound_dominance_ratio_Medium": "Medium Outbound Dominance",
    "outbound_dominance_ratio_Low":    "Low Outbound Dominance",
    "packet_rate_High":   "High Packet Rate",  "packet_rate_Medium": "Medium Packet Rate",
    "packet_rate_Low":    "Low Packet Rate",
    "bytes_per_packet_High":   "Large Bytes Per Packet",
    "bytes_per_packet_Medium": "Medium Bytes Per Packet",
    "bytes_per_packet_Low":    "Small Bytes Per Packet",
    "packet_asymmetry_High":   "Source-Dominant Traffic",
    "packet_asymmetry_Medium": "Balanced Traffic",
    "packet_asymmetry_Low":    "Dest-Dominant Traffic",
}

_SHORT_LABELS = {
    "sbytes": "Outbound Bytes",    "dbytes": "Inbound Bytes",
    "spkts":  "Source Packets",   "dpkts":  "Dest Packets",
    "dur":    "Session Duration", "rate":   "Transfer Rate",
    "sttl":   "Source TTL",       "dttl":   "Dest TTL",
    "sload":  "Source Load",      "dload":  "Dest Load",
    "outbound_dominance_ratio": "Outbound Dominance",
    "packet_rate":      "Packet Rate",
    "bytes_per_packet": "Bytes/Packet",
    "packet_asymmetry": "Traffic Asymmetry",
}

_CLUSTER_INSIGHTS = {
    3: ("Segment 3 consistently exhibits high inbound volume with compressed session "
        "durations, suggesting coordinated inbound burst or reconnaissance activity."),
    6: ("Segment 6 shows extreme outbound dominance with sustained session durations "
        "and minimal inbound response -- a behavioral signature strongly associated "
        "with data exfiltration."),
    9: ("Segment 9 demonstrates low packet rates over long session windows with "
        "source-dominant asymmetry, consistent with covert beacon or slow-scan "
        "command-and-control activity."),
}

_RISK_CONFIG = {
    3: {"tier": "HIGH RISK",     "badge_color": "#C0392B",
        "subtitle": "Inbound Burst / Reconnaissance Pattern",
        "badge_text_color": "white"},
    6: {"tier": "CRITICAL RISK", "badge_color": "#8B0000",
        "subtitle": "Exfiltration Candidate",
        "badge_text_color": "#FFD700"},
    9: {"tier": "HIGH RISK",     "badge_color": "#C0392B",
        "subtitle": "Covert Timing / Beacon Activity",
        "badge_text_color": "white"},
}


def _humanize(raw_items: str) -> str:
    """Convert comma-separated token labels to human-readable form."""
    parts = [t.strip() for t in raw_items.split(",")]
    return " + ".join(_TOKEN_LABELS.get(p, p) for p in parts)


def _wrap(text: str, width: int = 54) -> str:
    """Wrap a long rule string to at most `width` characters per line."""
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def _make_pattern_title(antecedents: str, consequents: str, max_len: int = 44) -> str:
    """Build a short, business-readable pattern title from raw token strings."""
    def _extract(token_str):
        labels = []
        for p in token_str.split(","):
            p = p.strip()
            for suffix in ("_High", "_Low", "_Medium"):
                if p.endswith(suffix):
                    base      = p[: -len(suffix)]
                    qualifier = suffix[1:]
                    alias     = _SHORT_LABELS.get(base, base.replace("_", " ").title())
                    labels.append(f"{qualifier} {alias}")
                    break
            else:
                labels.append(_SHORT_LABELS.get(p, p))
        return " + ".join(labels)

    ant_part = _extract(antecedents)
    con_part = _extract(consequents)
    title    = f"{ant_part} -> {con_part}"
    return title if len(title) <= max_len else ant_part[: max_len - 3] + "..."


def plot_top_rules(rules_by_cluster: dict,
                   priority_clusters: list,
                   top_n: int = 5,
                   plots_dir: str = "outputs/plots") -> None:
    """
    Generate one executive-ready SOC dashboard PNG per high-risk cluster.

    Layout (3-row GridSpec):
        Row 0 (58%) : Horizontal bar chart with lift highlighting
        Row 1 (13%) : Business risk insight paragraph
        Row 2 (29%) : Professional summary table

    Lift highlighting:
        lift > 7 : Gold bar, thicker, "Strong Association" tag
        lift > 5 : Bright cyan, slightly thicker
        baseline : Standard cluster color
    """
    import textwrap
    import matplotlib.gridspec as gridspec

    os.makedirs(plots_dir, exist_ok=True)

    _BASE_COLORS   = sns.color_palette("muted", 10)
    _LIFT5_COLOR   = "#00B4D8"
    _LIFT7_COLOR   = "#FFD700"
    _BADGE_FRAME   = dict(boxstyle="round,pad=0.5", linewidth=1.5)
    _INSIGHT_FRAME = dict(boxstyle="round,pad=0.6", facecolor="#0F2337",
                          edgecolor="#2A4060", linewidth=1.2, alpha=0.9)

    for cluster_id in priority_clusters:
        rules = rules_by_cluster.get(cluster_id)
        if rules is None or rules.empty:
            print(f"[INFO] Cluster {cluster_id}: no rules -- skipping.")
            continue

        top   = rules.head(top_n).copy().reset_index(drop=True)
        n     = len(top)
        lifts = top["lift"].values
        supps = top["support"].values
        confs = top["confidence"].values

        # Per-bar colors and heights based on lift tier
        bar_colors, bar_heights = [], []
        for lift in lifts:
            if lift > 7:
                bar_colors.append(_LIFT7_COLOR); bar_heights.append(0.62)
            elif lift > 5:
                bar_colors.append(_LIFT5_COLOR); bar_heights.append(0.58)
            else:
                bar_colors.append(_BASE_COLORS[cluster_id % len(_BASE_COLORS)])
                bar_heights.append(0.50)

        # Wrapped rule labels for y-axis
        wrapped_labels = [
            _wrap(f"{_humanize(row['antecedents'])} -> {_humanize(row['consequents'])}")
            for _, row in top.iterrows()
        ]

        # Short behavioral pattern titles for the table
        pattern_titles = [
            _make_pattern_title(row["antecedents"], row["consequents"])
            for _, row in top.iterrows()
        ]
        best_lift_idx = int(np.argmax(lifts))

        risk_cfg     = _RISK_CONFIG.get(cluster_id, {
            "tier": "ELEVATED RISK", "badge_color": "#8B6914",
            "subtitle": f"Anomalous Cluster {cluster_id}", "badge_text_color": "white"
        })
        insight_text = _CLUSTER_INSIGHTS.get(
            cluster_id,
            f"Segment {cluster_id} exhibits atypical behavioral patterns "
            "relative to dominant network traffic groups.",
        )

        # Figure layout
        fig_height = max(9, 1.85 * n + 5)
        fig = plt.figure(figsize=(15, fig_height), facecolor=_BG)
        gs  = gridspec.GridSpec(3, 1, figure=fig,
                                height_ratios=[0.58, 0.13, 0.29], hspace=0.5)

        # ── Row 0: Bar chart ──────────────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[0])
        ax_bar.set_facecolor("#1A2D42")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#2A4060")

        y_pos = list(range(n))
        bars  = ax_bar.barh(y_pos, lifts, height=bar_heights, color=bar_colors,
                            edgecolor="#FFFFFF", linewidth=0.7, alpha=0.92)

        x_max = max(lifts) * 1.28
        for bar, lift in zip(bars, lifts):
            lx = bar.get_width() + x_max * 0.018
            ax_bar.text(lx, bar.get_y() + bar.get_height() / 2,
                        f"{lift:.2f}", va="center", ha="left", fontsize=11,
                        color=_LIFT7_COLOR if lift > 7 else _TEXT, fontweight="bold")
            if lift > 7:
                ax_bar.text(lx + x_max * 0.08, bar.get_y() + bar.get_height() / 2,
                            "Strong Association", va="center", ha="left",
                            fontsize=8.5, color=_LIFT7_COLOR, style="italic")

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(wrapped_labels, fontsize=10, color=_TEXT, linespacing=1.35)
        ax_bar.set_xlabel("Lift Score", fontsize=11, color=_TEXT, labelpad=8)
        ax_bar.tick_params(axis="x", colors=_TEXT, labelsize=10)
        ax_bar.tick_params(axis="y", colors=_TEXT, length=0)
        ax_bar.set_xlim(0, x_max)
        ax_bar.grid(axis="x", color="#2A4060", linestyle="--", linewidth=0.6, alpha=0.6)

        ax_bar.set_title(
            f"Behavioral Segment {cluster_id}  --  {risk_cfg['subtitle']}\n"
            f"Top {n} Association Rules  |  Ranked by Lift Strength",
            fontsize=14, fontweight="bold", color=_TEXT, pad=20,
        )

        # Risk badge (top-right)
        ax_bar.text(0.985, 1.055, f"  {risk_cfg['tier']}  ",
                    transform=ax_bar.transAxes, ha="right", va="bottom",
                    fontsize=10, fontweight="bold", color=risk_cfg["badge_text_color"],
                    bbox={**_BADGE_FRAME, "facecolor": risk_cfg["badge_color"],
                          "edgecolor": risk_cfg["badge_text_color"]})

        # Framework subtitle (top-left)
        ax_bar.text(0.0, 1.055, "  Flagged: K-Means + DBSCAN dual-method framework  ",
                    transform=ax_bar.transAxes, ha="left", va="bottom",
                    fontsize=8.5, style="italic", color="#A0C8E8",
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "#0A2540",
                          "edgecolor": "#2A4060", "linewidth": 1.0})

        # ── Row 1: Insight paragraph ──────────────────────────────────────────
        ax_ins = fig.add_subplot(gs[1])
        ax_ins.set_facecolor(_BG)
        ax_ins.axis("off")
        wrapped_insight = "\n".join(textwrap.wrap(insight_text, width=105))
        ax_ins.text(0.5, 0.5, wrapped_insight, transform=ax_ins.transAxes,
                    ha="center", va="center", fontsize=10.5, style="italic",
                    color="#C8E6FF", linespacing=1.55, bbox=_INSIGHT_FRAME)

        # ── Row 2: Summary table ──────────────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[2])
        ax_tbl.set_facecolor(_BG)
        ax_tbl.axis("off")

        col_headers = ["Behavioral Pattern", "Support Rate", "Confidence Level", "Lift Strength"]
        table_data  = [
            [pattern_titles[i],
             f"{supps[i]*100:.1f}%",
             f"{confs[i]*100:.1f}%",
             f"{lifts[i]:.2f}"]
            for i in range(n)
        ]

        tbl = ax_tbl.table(cellText=table_data, colLabels=col_headers,
                           cellLoc="center", loc="center",
                           bbox=[0.0, 0.02, 1.0, 0.94])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        col_widths = [0.46, 0.18, 0.18, 0.18]
        for col_idx, width in enumerate(col_widths):
            for row_idx in range(n + 1):
                tbl[(row_idx, col_idx)].set_width(width)

        for col_idx in range(4):
            cell = tbl[(0, col_idx)]
            cell.set_facecolor("#0A3055")
            cell.set_text_props(color=_TEXT, fontweight="bold")
            cell.set_edgecolor("#2A4060")
            cell.set_height(0.22)

        for row_idx in range(1, n + 1):
            is_best = (row_idx - 1) == best_lift_idx
            row_bg  = "#0E2235" if is_best else ("#1A2D42" if row_idx % 2 == 0 else "#162233")
            for col_idx in range(4):
                cell = tbl[(row_idx, col_idx)]
                cell.set_facecolor(row_bg)
                cell.set_edgecolor("#2A4060")
                cell.set_height(0.20)
                props = {"color": _LIFT7_COLOR if is_best else _TEXT}
                if is_best:
                    props["fontweight"] = "bold"
                cell.set_text_props(**props)

        ax_tbl.set_title("Behavioral Pattern Summary",
                          fontsize=11, fontweight="bold", color=_TEXT, pad=8)

        save_path = os.path.join(plots_dir, f"association_cluster_{cluster_id}.png")
        plt.savefig(save_path, dpi=150, facecolor=_BG, bbox_inches="tight")
        plt.show()
        plt.close()
        print(f"[INFO] Saved: association_cluster_{cluster_id}.png")


# ── Label Evaluation Heatmap ──────────────────────────────────────────────────

def plot_cluster_attack_heatmap(crosstab, save_path: str) -> None:
    """
    Heatmap of K-Means cluster assignment vs ground-truth attack categories.

    Rows = K-Means clusters, Columns = attack_cat labels.
    Cell values = session count (log-scaled for readability).
    Used to validate that behavioral segments align with real attack types.
    """
    log_data = np.log1p(crosstab.values)

    fig, ax = plt.subplots(
        figsize=(max(10, len(crosstab.columns) * 1.2),
                 max(6, len(crosstab.index) * 0.75))
    )
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    im = ax.imshow(log_data, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns, rotation=38, ha="right",
                       fontsize=9, color=_TEXT)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels([f"Cluster {i}" for i in crosstab.index],
                       fontsize=9, color=_TEXT)

    for i in range(len(crosstab.index)):
        for j in range(len(crosstab.columns)):
            val = crosstab.values[i, j]
            if val > 0:
                txt_color = "black" if log_data[i, j] > 3 else _TEXT
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        fontsize=7, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("log(1 + sessions)", color=_TEXT, fontsize=9)
    cbar.ax.tick_params(colors=_TEXT)

    ax.set_title(
        "Cluster Assignment vs Ground-Truth Attack Categories\n"
        "Validation: Do Behavioral Segments Align with Real Attack Types?",
        fontsize=12, fontweight="bold", color=_TEXT, pad=12,
    )
    ax.set_xlabel("Attack Category (ground-truth label)", fontsize=10, color=_TEXT)
    ax.set_ylabel("K-Means Behavioral Segment", fontsize=10, color=_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A4060")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.show()
    plt.close()


# ── Recommender: Cluster Risk Heatmap ─────────────────────────────────────────

def plot_cluster_risk_heatmap(centroids: np.ndarray,
                               feature_names: list,
                               save_path: str,
                               top_n_features: int = 12) -> None:
    """
    Heatmap of K-Means centroid values across all clusters and top features.

    Each cell shows how far the cluster centroid deviates (in std-devs) from
    the overall mean for that feature.  Red = HIGH, blue = LOW.

    Business use: SOC managers can scan which clusters dominate on which
    features — instantly surfacing the behavioral fingerprint of each segment.

    Parameters:
        centroids      : (k, n_features) centroid matrix from K-Means
        feature_names  : Feature column names matching centroid columns
        save_path      : Output PNG path
        top_n_features : Max features to show (ranked by overall variance)
    """
    # Pick top features by mean absolute centroid value (most discriminating)
    mean_abs = np.abs(centroids).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n_features]
    sub_data = centroids[:, top_idx]
    sub_feat = [feature_names[i] for i in top_idx]

    k = centroids.shape[0]
    n_cols = sub_data.shape[1]     # actual number shown (may be < top_n_features)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.9), max(5, k * 0.7)))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    im = ax.imshow(sub_data, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=2.5)

    # Annotate cells
    for i in range(k):
        for j in range(n_cols):
            val = sub_data[i, j]
            txt_color = "black" if abs(val) > 1.5 else _TEXT
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sub_feat, rotation=40, ha="right",
                       fontsize=9, color=_TEXT)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Cluster {i}" for i in range(k)],
                       fontsize=9, color=_TEXT)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Centroid deviation (std-devs from mean)",
                   color=_TEXT, fontsize=9)
    cbar.ax.tick_params(colors=_TEXT)

    ax.set_title(
        "Cluster Behavioral Risk Fingerprint\n"
        "Centroid Feature Deviations Across All Behavioral Segments",
        fontsize=13, fontweight="bold", color=_TEXT, pad=14,
    )
    ax.set_xlabel("Network Feature", fontsize=10, color=_TEXT)
    ax.set_ylabel("K-Means Behavioral Segment", fontsize=10, color=_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A4060")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.show()
    plt.close()


# ── Search: Results Plot ───────────────────────────────────────────────────────

def plot_search_results(q1_results, q2_results,
                        top_k: int, save_path: str) -> None:
    """
    Side-by-side horizontal bar charts for the two SOC search queries.

    Left panel  (Q1): "Sessions similar to Cluster 6 centroid profile"
    Right panel (Q2): "Sessions matching exfiltration feature signature"

    Bar length = similarity score.  Bars are coloured by cluster assignment
    so analysts can immediately see whether results span clusters or are
    concentrated in one high-risk segment.

    Parameters:
        q1_results : DataFrame from search Q1 (cluster-centroid query)
        q2_results : DataFrame from search Q2 (feature-profile query)
        top_k      : Number of results displayed
        save_path  : Output PNG path
    """
    palette = sns.color_palette("tab10", 10)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, top_k * 0.65)))
    fig.patch.set_facecolor(_BG)

    queries = [
        (q1_results, "Q1 — Cluster-Centroid Match\n(Cluster 6 Profile)"),
        (q2_results, "Q2 — Exfiltration Signature Match\n(Feature Profile Query)"),
    ]

    for ax, (df, title) in zip(axes, queries):
        ax.set_facecolor("#1A2D42")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A4060")

        if df.empty:
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                    color=_TEXT, fontsize=12, transform=ax.transAxes)
            ax.set_title(title, fontsize=11, color=_TEXT, fontweight="bold")
            continue

        scores  = df["similarity_score"].values
        labels  = df["cluster"].values
        y_pos   = range(len(scores))
        colors  = [palette[int(lb) % len(palette)] for lb in labels]
        y_labels = [f"Session {int(df.iloc[i]['session_idx']):,}  "
                    f"[C{int(labels[i])}]"
                    for i in range(len(df))]

        bars = ax.barh(list(y_pos), scores, color=colors,
                       edgecolor="white", linewidth=0.5,
                       height=0.6, alpha=0.88)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}",
                    va="center", ha="left", fontsize=9,
                    color=_YELLOW, fontweight="bold")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(y_labels, fontsize=9, color=_TEXT)
        ax.set_xlabel("Cosine Similarity Score", fontsize=10, color=_TEXT,
                      labelpad=6)
        ax.tick_params(axis="x", colors=_TEXT, labelsize=9)
        ax.tick_params(axis="y", colors=_TEXT, length=0)
        ax.set_xlim(0, 1.12)
        ax.grid(axis="x", color="#2A4060", linestyle="--",
                linewidth=0.6, alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=_TEXT, pad=10)

    fig.suptitle(
        "Behavioral Similarity Search — SOC Query Results\n"
        "Ranked by Cosine Similarity Score  |  [C#] = Cluster Assignment",
        fontsize=13, fontweight="bold", color=_TEXT, y=1.02,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.show()
    plt.close()


# ── Isolation Forest: SOC Anomaly Dashboard ───────────────────────────────────

def plot_if_results(if_result: dict, km_result: dict, save_path: str) -> None:
    """
    3-panel SOC dashboard for Isolation Forest anomaly results.

    Panel 1 (top-left)  : Anomaly score distribution — normal vs anomaly
    Panel 2 (top-right) : Per-cluster anomaly rate (% flagged by IF)
    Panel 3 (bottom)    : Top-20 SOC queue — anomaly scores per session,
                          dual-flagged sessions highlighted

    Parameters:
        if_result : Output dict from run_isolation_forest()
        km_result : K-Means result dict (for cluster colour palette)
        save_path : Output PNG path
    """
    import matplotlib.gridspec as gridspec

    scores      = if_result["anomaly_score"]
    flags       = if_result["anomaly_flag"]
    soc_queue   = if_result["soc_queue"]
    summary_df  = if_result["summary_df"]

    palette = sns.color_palette("tab10", 10)
    n_anom  = if_result["n_anomalies"]
    n_total = len(scores)

    fig = plt.figure(figsize=(16, 10), facecolor=_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Score distribution ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1A2D42")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#2A4060")

    normal_scores  = scores[flags == 0]
    anomaly_scores = scores[flags == 1]

    ax1.hist(normal_scores,  bins=60, color=_ACCENT, alpha=0.75,
             label=f"Normal  ({n_total - n_anom:,})", density=True)
    ax1.hist(anomaly_scores, bins=40, color=_RED,    alpha=0.85,
             label=f"Anomaly ({n_anom:,})", density=True)
    ax1.axvline(0, color=_YELLOW, linewidth=1.5, linestyle="--",
                label="Decision boundary")
    ax1.set_xlabel("Anomaly Score (lower = more anomalous)",
                   fontsize=10, color=_TEXT)
    ax1.set_ylabel("Density", fontsize=10, color=_TEXT)
    ax1.tick_params(colors=_TEXT)
    ax1.legend(fontsize=8.5, facecolor=_BG, labelcolor=_TEXT)
    ax1.set_title("Anomaly Score Distribution\nNormal vs Flagged Sessions",
                  fontsize=11, fontweight="bold", color=_TEXT, pad=10)

    # ── Panel 2: Per-cluster anomaly rate ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1A2D42")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#2A4060")

    cids   = summary_df["cluster"].values
    rates  = summary_df["if_anomaly_rate_%"].values
    colors = [palette[int(c) % len(palette)] for c in cids]

    bars = ax2.bar(cids, rates, color=colors, edgecolor="white",
                   linewidth=0.6, alpha=0.88)
    for bar, rate in zip(bars, rates):
        if rate > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     f"{rate:.1f}%", ha="center", va="bottom",
                     fontsize=8.5, color=_TEXT, fontweight="bold")

    ax2.set_xlabel("K-Means Cluster", fontsize=10, color=_TEXT)
    ax2.set_ylabel("Anomaly Rate (%)", fontsize=10, color=_TEXT)
    ax2.set_xticks(cids)
    ax2.tick_params(colors=_TEXT)
    ax2.set_title("Isolation Forest Anomaly Rate per Cluster\n"
                  "% of Sessions Flagged as Anomalous",
                  fontsize=11, fontweight="bold", color=_TEXT, pad=10)

    # ── Panel 3 (full width): SOC priority queue top-20 ──────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#1A2D42")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#2A4060")

    top20 = soc_queue.head(20).reset_index(drop=True)
    y_pos = list(range(len(top20)))
    bar_colors = [
        "#8B0000" if "dual" in top20.columns and top20.iloc[i].get("dual", 0)
        else _RED
        for i in range(len(top20))
    ]
    # All SOC queue bars are anomalies — colour by cluster
    bar_colors = [palette[int(top20.iloc[i]["km_cluster"]) % len(palette)]
                  for i in range(len(top20))]

    ax3.barh(y_pos, top20["anomaly_score"].abs(), color=bar_colors,
             edgecolor="white", linewidth=0.5, height=0.65, alpha=0.88)

    ylabels = [f"#{i+1}  Session {int(top20.iloc[i]['session_idx']):,}  "
               f"[C{int(top20.iloc[i]['km_cluster'])}]"
               for i in range(len(top20))]
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(ylabels, fontsize=9, color=_TEXT)
    ax3.set_xlabel("|Anomaly Score|  (higher = more suspicious)",
                   fontsize=10, color=_TEXT, labelpad=6)
    ax3.tick_params(axis="x", colors=_TEXT)
    ax3.tick_params(axis="y", colors=_TEXT, length=0)
    ax3.grid(axis="x", color="#2A4060", linestyle="--",
             linewidth=0.6, alpha=0.5)
    ax3.set_title(
        f"SOC Priority Queue — Top {len(top20)} Highest-Risk Sessions\n"
        f"Total Flagged: {n_anom:,} ({if_result['anomaly_pct']}%)  "
        f"|  Dual-flagged (IF + DBSCAN): {if_result['dual_flagged']:,}  "
        f"[Colours = K-Means Cluster]",
        fontsize=11, fontweight="bold", color=_TEXT, pad=10,
    )

    fig.suptitle(
        "Isolation Forest Anomaly Detection — SOC Dashboard\n"
        "Global Behavioral Outlier Analysis  |  UNSW-NB15 Dataset",
        fontsize=14, fontweight="bold", color=_TEXT, y=1.01,
    )


    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.show()
    plt.close()


# ── Association Rule Mining Visualisation ──────────────────────────────────────

def plot_top_rules(
    rules: list,
    top_n: int = 20,
    save_path: str = "outputs/plots/association_rules.png",
) -> None:
    """
    Horizontal bar chart of the top-N association rules ranked by lift.

    Parameters
    ----------
    rules     : list of dicts from run_apriori() — each dict has
                  antecedent (frozenset), consequent (frozenset),
                  support (float), confidence (float), lift (float)
    top_n     : number of rules to display (default 20)
    save_path : output PNG path
    """
    if not rules:
        print("  [WARN] No association rules to plot.")
        return

    sorted_rules = sorted(rules, key=lambda r: r["lift"], reverse=True)[:top_n]
    if not sorted_rules:
        return

    labels = []
    lifts  = []
    confs  = []
    for r in sorted_rules:
        ant = " AND ".join(sorted(r["antecedent"]))
        con = " AND ".join(sorted(r["consequent"]))
        label = f"{ant}  →  {con}"
        if len(label) > 80:
            label = label[:77] + "..."
        labels.append(label)
        lifts.append(r["lift"])
        confs.append(r["confidence"])

    # Reverse so highest lift is at the top
    labels = labels[::-1]
    lifts  = lifts[::-1]
    confs  = confs[::-1]

    fig_height = max(6, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    _apply_dark_bg(fig, ax)

    bars = ax.barh(range(len(labels)), lifts, color=_ACCENT, alpha=0.85,
                   edgecolor="none")

    # Colour by confidence: low conf = yellow, high conf = accent
    for bar, conf in zip(bars, confs):
        bar.set_color(_ACCENT if conf >= 0.85 else _YELLOW)
        bar.set_alpha(0.85)

    # Lift value labels
    for i, (lift, conf) in enumerate(zip(lifts, confs)):
        ax.text(lift + 0.02, i,
                f"lift={lift:.2f}  conf={conf:.2f}",
                va="center", ha="left", color=_TEXT, fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color=_TEXT)
    ax.set_xlabel("Lift  (confidence / P(consequent))", color=_TEXT, fontsize=10)
    ax.set_title(
        f"Top {len(labels)} Association Rules — Ranked by Lift\n"
        "Behavioral Co-occurrence Patterns  |  UNSW-NB15",
        color=_TEXT, fontsize=12, fontweight="bold",
    )

    # Legend
    accent_patch = mpatches.Patch(color=_ACCENT, label="High confidence (≥ 0.85)")
    yellow_patch = mpatches.Patch(color=_YELLOW, label="Moderate confidence (< 0.85)")
    ax.legend(handles=[accent_patch, yellow_patch], loc="lower right",
              facecolor=_BG, labelcolor=_TEXT, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=_BG, bbox_inches="tight")
    plt.show()
    plt.close()


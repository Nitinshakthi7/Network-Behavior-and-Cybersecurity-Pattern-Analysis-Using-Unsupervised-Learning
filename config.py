"""
config.py
---------
Central configuration for the Unsupervised Behavioral Network Traffic Analysis Framework.
All tunable parameters live here. No hardcoding in source modules.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "UNSW_NB15_training-set.csv")
OUTPUTS_DIR = "outputs"
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR  = os.path.join(OUTPUTS_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# ── Dataset ───────────────────────────────────────────────────────────────────
# Columns excluded from all model training (used only for post-hoc validation)
LABEL_COLS = ["label", "attack_cat"]

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Feature Engineering ───────────────────────────────────────────────────────
# Set to False to skip engineered features and use raw features only
FEATURE_ENGINEERING_ENABLED = True

# ── K-Means ───────────────────────────────────────────────────────────────────
# Range of K values to evaluate (inclusive on both ends)
KMEANS_K_MIN = 2
KMEANS_K_MAX = 10

# Number of rows sampled for silhouette score computation (performance)
# Full dataset is always used for model fitting
KMEANS_SILHOUETTE_SAMPLE_SIZE = 10_000

# ── DBSCAN ────────────────────────────────────────────────────────────────────
# Maximum Euclidean distance (in standardised space) between two neighbours
DBSCAN_EPS = 1.5

# Minimum points within eps radius to classify a point as a core point
DBSCAN_MIN_SAMPLES = 10

# ── Visualization ─────────────────────────────────────────────────────────────
# Number of rows sampled for scatter plot (performance)
SCATTER_SAMPLE_SIZE = 10_000

# Feature names to use for the 2D scatter projection
# Must exist in the feature matrix after preprocessing + engineering
SCATTER_FEAT_X = "sbytes"
SCATTER_FEAT_Y = "dbytes"

# ── Reporting ─────────────────────────────────────────────────────────────────
# Clusters with fewer than this fraction of total records are flagged as rare
RARE_CLUSTER_THRESHOLD = 0.01   # 1 % of total traffic

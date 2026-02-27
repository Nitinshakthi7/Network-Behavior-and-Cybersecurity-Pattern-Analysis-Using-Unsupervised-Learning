# Network Behavior & Cybersecurity Pattern Analysis Using Unsupervised Learning

> **Unsupervised Behavioral Threat Detection Framework**
> Built for enterprise Security Operations Centers (SOC) that need to detect unknown threats at scale — without relying on labeled attack data or signature rules.

---

## 🏢 Business Context

Modern enterprise networks generate hundreds of thousands of sessions daily. Signature-based detection systems fail against zero-day exploits, low-and-slow intrusions, and insider threats — because those threats have no prior signature to match.

This framework takes a different approach: **learn what normal looks like, then surface everything that doesn't fit.**

By applying unsupervised behavioral segmentation to raw network telemetry, the framework:

- Partitions all traffic into **10 distinct behavioral profiles** — without using attack labels
- Automatically flags **rare behavioral segments** for priority investigation
- Isolates **6,983 sessions (3.98%)** as density anomalies — reducing analyst workload by **96%**
- Provides a ranked SOC investigation queue, from critical exfiltration candidates down to routine monitoring

The system does not replace analysts. It tells them exactly where to look first.

---

> **University Machine Learning Project — 2nd Year Digital Technology**
> Dataset: UNSW-NB15 Training Set | Phase: Clustering (Interim)

---

## 📌 Project Overview

This project applies **unsupervised machine learning** to the UNSW-NB15 network intrusion dataset to identify distinct network traffic behaviour groups and surface anomalous sessions — without using labels during training.

Labels (`label`, `attack_cat`) exist in the dataset but are **strictly excluded from all model training**. They are reserved for post-hoc validation only.

---

## ✅ What Has Been Implemented (Current Phase)

### 1. Data Preprocessing (`src/preprocessing.py`)

- Loads UNSW-NB15 CSV dataset
- Handles missing values (median for numerical, mode for categorical)
- Separates and excludes label columns from training
- Label encodes categorical features: `proto`, `service`, `state`
- Applies StandardScaler — normalises all features to mean=0, std=1

### 2. Feature Engineering (`src/feature_engineering.py`)

Four domain-informed behavioral indicators computed from raw features:

| Feature                    | Formula                             | Security Interpretation                              |
| -------------------------- | ----------------------------------- | ---------------------------------------------------- |
| `outbound_dominance_ratio` | `sbytes / (sbytes + dbytes)`        | Near 1.0 = potential exfiltration posture            |
| `packet_rate`              | `spkts / dur`                       | Very high = scan/flood; regular = C2 beacon          |
| `bytes_per_packet`         | `sbytes / spkts`                    | Small = scan probe; large = bulk transfer            |
| `packet_asymmetry`         | `(spkts - dpkts) / (spkts + dpkts)` | +1 = source-dominant (DoS/exfil); −1 = dest-dominant |

Toggle via `FEATURE_ENGINEERING_ENABLED` in `config.py`.

### 3. K-Means Clustering (`src/clustering/kmeans.py`)

- Tests K values from 2 to 10 (configurable in `config.py`)
- Computes Silhouette Score for each K (on a 10,000-row sample)
- Automatically selects best K (highest silhouette score)
- **Result: Best K = 10 | Silhouette Score = 0.3529**
- Initialisation: `k-means++` with fixed `random_state=42`

### 4. DBSCAN Clustering (`src/clustering/dbscan.py`)

- Parameters: `eps=1.5`, `min_samples=10`
- Identifies density-based clusters without pre-specifying K
- Labels outlier sessions as noise (`-1`)
- **Result: 113 clusters | 6,983 noise points (3.98%)**
- Parallelised with `n_jobs=-1`

### 5. Evaluation (`src/clustering/evaluation.py`)

- Silhouette Score computation with sampling
- Cluster size distribution reporting
- K-Means centroid analysis — top 5 high/low features per cluster
- DBSCAN noise point reporting

### 6. Visualization (`src/visualization.py`)

Four plots saved to `outputs/plots/`:

| Plot File                  | Description                                                   |
| -------------------------- | ------------------------------------------------------------- |
| `silhouette_vs_k.png`      | Silhouette score vs K — best K highlighted                    |
| `kmeans_cluster_sizes.png` | Bar chart of points per K-Means cluster                       |
| `dbscan_summary.png`       | DBSCAN summary: clusters, noise, clustered points             |
| `kmeans_scatter.png`       | 2D scatter of K-Means clusters (sbytes vs dbytes, 10k sample) |

### 7. Reporting (`src/reporting.py`)

- `outputs/reports/summary.txt` — full text summary of clustering results
- `outputs/reports/cluster_summary.csv` — per-cluster mean feature values

### 8. Central Configuration (`config.py`)

All parameters in one place — K range, DBSCAN eps/min_samples, file paths, sampling sizes, rare cluster threshold.

---

## 🔜 Not Yet Implemented (Planned — Future Phases)

- [ ] Anomaly Detection — Isolation Forest (`src/anomaly/`)
- [ ] Association Rule Mining — Apriori / FP-Growth (`src/association/`)
- [ ] Policy Recommendation Engine (`src/recommendation/`)
- [ ] Interactive Session Query System (`src/search/`)
- [ ] Formal evaluation against attack labels (`attack_cat`)
- [ ] k-distance plot for principled eps selection

---

## 📁 Project Structure

```
.
├── config.py                        ← All parameters centralised here
├── main.py                          ← Entry point — runs full pipeline
├── data/
│   └── UNSW_NB15_training-set.csv   ← Dataset (not included in repo)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── utils.py
│   ├── visualization.py
│   ├── reporting.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── kmeans.py
│   │   ├── dbscan.py
│   │   └── evaluation.py
│   ├── anomaly/                     ← Placeholder (Phase 2)
│   ├── association/                 ← Placeholder (Phase 3)
│   ├── recommendation/              ← Placeholder (Phase 4)
│   └── search/                      ← Placeholder (Phase 4)
├── outputs/
│   ├── plots/                       ← Generated plot images
│   ├── models/                      ← Saved models (future)
│   └── reports/                     ← summary.txt + cluster_summary.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Network-Behavior-and-Cybersecurity-Pattern-Analysis-Using-Unsupervised-Learning.git
cd Network-Behavior-and-Cybersecurity-Pattern-Analysis-Using-Unsupervised-Learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download the UNSW-NB15 training set from:

> https://research.unsw.edu.au/projects/unsw-nb15-dataset

Place the file at:

```
data/UNSW_NB15_training-set.csv
```

### 4. Run the pipeline

```bash
python main.py
```

---

## 📊 Key Results (Clustering Phase)

| Method                       | Result                                      |
| ---------------------------- | ------------------------------------------- |
| K-Means Best K               | 10                                          |
| K-Means Silhouette Score     | 0.3529                                      |
| Auto-flagged rare clusters   | 3 (Clusters 3, 6, 9)                        |
| Smallest cluster (Cluster 6) | 82 points — critical exfiltration candidate |
| DBSCAN Clusters Found        | 113                                         |
| DBSCAN Noise Points          | 6,983 (3.98% of data)                       |

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

## ⚠️ Notes

- The `data/` folder is excluded from version control (see `.gitignore`). Download the dataset separately.
- All models use `random_state=42` for reproducibility.
- Labels are never passed to any model. Unsupervised only.
- DBSCAN uses `n_jobs=-1` to parallelise across all CPU cores.

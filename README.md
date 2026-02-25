# Network Behavior & Cybersecurity Pattern Analysis Using Unsupervised Learning

> **University Machine Learning Project — 2nd Year Digital Transformation**
> Dataset: UNSW-NB15 Training Set | Phase: Clustering (Interim)

---

## 📌 Project Overview

This project applies **unsupervised machine learning** techniques to the UNSW-NB15 network intrusion dataset to identify distinct network traffic behaviour groups and surface anomalous sessions — without using labels during training.

Labels (`label`, `attack_cat`) exist in the dataset but are **strictly excluded from all model training**. They are reserved for post-hoc validation only.

---

## ✅ What Has Been Implemented (Current Phase)

### 1. Data Preprocessing (`src/preprocessing.py`)

- Loads UNSW-NB15 CSV dataset
- Handles missing values (median for numerical, mode for categorical)
- Separates and excludes label columns from training
- Label encodes categorical features: `proto`, `service`, `state`
- Applies StandardScaler — normalises all 43 features to mean=0, std=1

### 2. K-Means Clustering (`src/clustering.py`)

- Tests K values from 2 to 8
- Computes Silhouette Score for each K (on a 10,000-row sample)
- Automatically selects best K (highest silhouette score)
- **Result: Best K = 7 | Silhouette Score = 0.3476**
- Initialisation: `k-means++` with fixed `random_state=42`

### 3. DBSCAN Clustering (`src/clustering.py`)

- Parameters: `eps=1.5`, `min_samples=10`
- Identifies density-based clusters without pre-specifying K
- Labels outlier points as noise (`-1`)
- **Result: 107 clusters | 6,408 noise points (3.65%)**
- Parallelised with `n_jobs=-1`

### 4. Evaluation (`src/evaluation.py`)

- Silhouette Score computation with sampling
- Cluster size distribution printing
- K-Means centroid analysis — top 5 high/low features per cluster
- DBSCAN noise point reporting

### 5. Visualization (`src/visualization.py`)

Four plots generated and saved to `plots/`:

| Plot File                  | Description                                                   |
| -------------------------- | ------------------------------------------------------------- |
| `silhouette_vs_k.png`      | Silhouette score vs K — best K highlighted                    |
| `kmeans_cluster_sizes.png` | Bar chart of points per K-Means cluster                       |
| `dbscan_summary.png`       | DBSCAN summary: clusters, noise, clustered points             |
| `kmeans_scatter.png`       | 2D scatter of K-Means clusters (sbytes vs dbytes, 10k sample) |

---

## 🔜 Not Yet Implemented (Planned — Future Phases)

- [ ] Anomaly Detection (Isolation Forest — after coursework coverage)
- [ ] Formal evaluation against attack labels (`attack_cat`)
- [ ] DBSCAN parameter sensitivity analysis (k-distance plot for eps selection)
- [ ] Dimensionality reduction visualization (if studied in coursework)
- [ ] Feature importance / cluster semantic labeling

---

## 📁 Project Structure

```
.
├── data/
│   └── UNSW_NB15_training-set.csv      ← Dataset (not included in repo)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                ← Data loading & preprocessing
│   ├── clustering.py                   ← K-Means & DBSCAN
│   ├── evaluation.py                   ← Silhouette score & centroid analysis
│   └── visualization.py               ← All 4 plots
├── plots/                              ← Generated plot images
├── Refrences Docs/                     ← University assignment references
├── main.py                             ← Entry point — runs full pipeline
├── generate_presentation.py           ← Generates interim .pptx presentation
├── Interim_Presentation.pptx          ← Generated presentation file
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

Place the file as:

```
data/UNSW_NB15_training-set.csv
```

### 4. Run the pipeline

```bash
python main.py
```

### 5. (Optional) Regenerate the presentation

```bash
python generate_presentation.py
```

---

## 📊 Key Results (Clustering Phase)

| Method                       | Result                             |
| ---------------------------- | ---------------------------------- |
| K-Means Best K               | 7                                  |
| K-Means Silhouette Score     | 0.3476                             |
| Smallest cluster (Cluster 6) | 89 points — highly unusual traffic |
| DBSCAN Clusters Found        | 107                                |
| DBSCAN Noise Points          | 6,408 (3.65% of data)              |
| DBSCAN Silhouette Score      | 0.0332                             |

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
python-pptx
```

---

## ⚠️ Notes

- The `data/` folder is excluded from version control (see `.gitignore`). Download the dataset separately.
- `plots/` contains generated images — these may be committed as evidence of results.
- All models use `random_state=42` for reproducibility.
- Labels are never passed to any model. Unsupervised only.

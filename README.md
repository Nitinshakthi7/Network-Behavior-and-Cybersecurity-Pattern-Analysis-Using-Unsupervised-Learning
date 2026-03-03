# Behavioral Threat Detection Framework

## Unsupervised Machine Learning for Network Traffic Analysis

### Overview

This project implements a behavioral threat detection framework using unsupervised machine learning techniques on the UNSW-NB15 benchmark dataset.

The objective is to:

- Segment network traffic based on behavioral patterns
- Detect anomalous sessions without relying on attack labels
- Demonstrate how such a system could assist a Security Operations Center (SOC) in prioritizing investigations

**All modeling is performed in a fully unsupervised manner.** Ground-truth labels are used only for post-hoc validation, not for training.

---

### Dataset

- **Dataset:** UNSW-NB15
- **Source:** Australian Centre for Cybersecurity, UNSW Canberra
- **Records Used:** 175,341 network flow sessions (Training set)
- **Features:** 45 raw features

> ⚠️ **Important:**
> This dataset is a research benchmark generated in a controlled laboratory environment using traffic simulation tools. It is not real corporate production traffic.
> The framework demonstrates how behavioral segmentation and anomaly detection would operate if deployed in a real enterprise environment.

---

### Project Objectives

- Segment network sessions using behavioral similarity
- Detect density-based anomalies
- Discover co-occurring behavioral patterns across attributes
- Detect globally extreme outliers
- Combine multiple anomaly signals to create a risk-prioritized output
- Validate behavioral clusters against known attack categories

---

### Methods Used

#### 1. Feature Engineering

Selected core session-level features:

- Bytes sent / received
- Packet counts
- Duration
- Protocol / service / state

Engineered behavioral features:

- **Outbound dominance ratio:** Proportion of traffic originating from the source
- **Packet rate:** Packets per second
- **Bytes per packet:** Average payload size
- **Packet asymmetry:** Imbalance between source and destination packets

_These features convert raw traffic into behavioral signals suitable for clustering._

#### 2. K-Means Clustering

- **Purpose:** Behavioral segmentation.
- K selected using Silhouette Score (K = 10)
- StandardScaler applied prior to clustering
- Euclidean distance in standardized feature space
- Deterministic initialization (k-means++)
- **Outcome:** 10 behavioral clusters. Majority represent normal traffic. Rare clusters (<2%) flagged for further investigation.

#### 3. DBSCAN

- **Purpose:** Density-based anomaly detection.
- `eps = 1.5`, `min_samples = 10`
- ~4% of sessions flagged as density anomalies.
- Detects locally unusual sessions that do not belong to any dense behavioral region.

#### 4. Association Rule Mining (Apriori)

- **Purpose:** Discovering co-occurring behavioral patterns.
- Mines frequent itemsets across all discretised traffic features globally.
- Surfaces rules ranked by _lift_ (where conditional probability exceeds expected random occurrence).
- Extracts explainable context for SOC analysts (e.g., `proto_tcp AND packet_rate_high → outbound_dominance_ratio_high`).

#### 5. Isolation Forest

- **Purpose:** Global anomaly detection.
- Tree-based outlier detection with no distance assumptions.
- ~4% sessions flagged as global anomalies.
- Detects sessions with extreme feature profiles.

#### 6. Multi-Method Consensus

Sessions flagged by:

- Rare K-Means cluster
- DBSCAN noise
- Isolation Forest anomaly

These are ranked highest for investigation. This layered approach reduces false positives compared to single-method detection.

---

### Validation Strategy

Ground-truth attack labels were **not** used during modeling.
They were used only for:

- Post-hoc cluster validation (Purity and Adjusted Rand Index)
- Heatmap analysis (Cluster × Attack Category)

**Result:** Certain clusters showed significant enrichment for specific attack types, indicating meaningful behavioral grouping without relying on signatures.

---

### Project Structure

```text
Network-Behavior-and-Cybersecurity-Pattern-Analysis-Using-Unsupervised-Learning/
├── main.py                     # Main execution pipeline
├── config.py                   # Configuration parameters
├── src/
│   ├── preprocessing.py        # Data loading and cleaning
│   ├── feature_engineering.py  # Behavioral signal creation
│   ├── clustering/
│   │   ├── kmeans.py           # K-Means segmentation
│   │   ├── dbscan.py           # Density-based outlier detection
│   │   └── evaluation.py       # Centroid stats and cluster distribution
│   ├── anomaly/
│   │   └── isolation_forest.py # Global anomaly scoring
│   ├── association/
│   │   ├── apriori.py          # Custom Apriori implementation
│   │   └── __init__.py
│   ├── visualization.py        # 6 core plots
│   ├── reporting.py            # Text and CSV summary generation
│   ├── evaluation_labels.py    # Post-hoc label validation
│   └── utils.py
├── outputs/
│   ├── plots/                  # Generated PNG visualisations
│   ├── reports/                # Generated CSVs and txt reports
│   └── models/                 # Directory for serialised models (e.g. .pkl)
├── requirements.txt            # Python dependencies
└── README.md
```

---

### How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Place Dataset:**
   Place the UNSW-NB15 training CSV in the root or `data/` directory (update the path in `preprocessing.py` or your execution script).

3. **Run Pipeline:**
   ```bash
   python main.py
   ```

Outputs will be saved in `outputs/plots/`, `outputs/reports/`, and the terminal console. All generated association rules will be saved to `outputs/reports/association_rules.csv`.

---

### Business Perspective

**If deployed in an enterprise SOC:**

_Without this system:_

- Analysts triage alerts manually
- Signature-based detection misses unknown/zero-day attacks
- Investigation priority is unclear

_With this system:_

- Traffic is segmented behaviorally
- Anomalies are automatically surfaced via dual-consensus (DBSCAN + Isolation Forest)
- Sessions are risk-ranked
- Mean Time to Detect (MTTD) involves prioritizing high-risk behavioral clusters

---

### Limitations

- Dataset is simulated benchmark data.
- Euclidean distance may degrade in high-dimensional space.
- DBSCAN is sensitive to the `eps` parameter.
- Model requires periodic retraining to handle concept drift.
- No precision/recall metrics during inference (due to strictly unsupervised setting).

### Future Work

- HDBSCAN for improved density detection without explicit `eps`.
- Temporal sequence modeling (e.g., beacon detection, lateral movement tracing).
- Feedback loop from analyst-confirmed incidents to weight features dynamically.
- Real-time streaming integration via Kafka/Spark.
- Evaluation on real-world enterprise NetFlow or PCAP data.

---

### Key Takeaways

- Fully unsupervised behavioral modeling
- Multi-layer anomaly detection (Isolation Forest + DBSCAN)
- Explainable pattern discovery (Apriori rules)
- Interpretable clustering and segmentation
- Business-relevant risk prioritization
- Academic integrity maintained (labels strictly withheld from training)

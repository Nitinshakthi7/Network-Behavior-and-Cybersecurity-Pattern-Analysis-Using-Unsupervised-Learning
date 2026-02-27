"""
feature_engineering.py
-----------------------
Constructs domain-informed features from raw network session attributes.

All engineered features are directly interpretable in a cybersecurity context.
Feature engineering is configurable — set FEATURE_ENGINEERING_ENABLED=False
in config.py to run the pipeline on raw features only.

Engineered features:
    outbound_dominance_ratio  — fraction of total bytes sent outbound
    packet_rate               — packets transmitted per second
    bytes_per_packet          — average payload size per source packet
    packet_asymmetry          — directional imbalance in packet counts

Division-by-zero is handled via utils.safe_divide (fills with 0.0).
"""

import pandas as pd
import numpy as np
from src.utils import safe_divide


# ── Feature definitions ───────────────────────────────────────────────────────

def outbound_dominance_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Fraction of total bytes transmitted in the outbound (source) direction.

    Formula: sbytes / (sbytes + dbytes)

    Interpretation:
        ~0.5  → symmetric, bidirectional session (normal browsing)
        → 1.0 → almost all bytes sent outbound (potential exfiltration or scan)
        → 0.0 → mostly inbound bytes (possible large download or C2 response)
    """
    total = df["sbytes"] + df["dbytes"]
    return pd.Series(safe_divide(df["sbytes"].values, total.values), index=df.index)


def packet_rate(df: pd.DataFrame) -> pd.Series:
    """
    Source packets transmitted per second of session duration.

    Formula: spkts / dur

    Interpretation:
        High packet rate → scanning, flood-type behaviour
        Low packet rate  → slow, stealthy, or long-idle session
        dur = 0 filled with 0.0 (instantaneous session)
    """
    return pd.Series(safe_divide(df["spkts"].values, df["dur"].values), index=df.index)


def bytes_per_packet(df: pd.DataFrame) -> pd.Series:
    """
    Average payload size per source packet (bytes).

    Formula: sbytes / spkts

    Interpretation:
        Small value → many tiny packets (SYN scan, keepalive traffic)
        Large value → few large packets (bulk file transfer, exfiltration)
    """
    return pd.Series(safe_divide(df["sbytes"].values, df["spkts"].values), index=df.index)


def packet_asymmetry(df: pd.DataFrame) -> pd.Series:
    """
    Directional imbalance in packet count between source and destination.

    Formula: (spkts - dpkts) / (spkts + dpkts)

    Range: -1.0 (all packets from destination) to +1.0 (all packets from source)

    Interpretation:
        ≈  0  → balanced bidirectional exchange (normal session)
        → +1  → source-heavy session (potential DoS, scan, or exfiltration)
        → -1  → destination-heavy session (large response body, C2 polling)
    """
    total = df["spkts"] + df["dpkts"]
    diff  = df["spkts"] - df["dpkts"]
    return pd.Series(safe_divide(diff.values, total.values), index=df.index)


# ── Orchestrator ──────────────────────────────────────────────────────────────

ENGINEERED_FEATURE_MAP = {
    "outbound_dominance_ratio": outbound_dominance_ratio,
    "packet_rate":              packet_rate,
    "bytes_per_packet":         bytes_per_packet,
    "packet_asymmetry":         packet_asymmetry,
}


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features and append them to the DataFrame.

    Required raw columns:
        sbytes, dbytes, spkts, dpkts, dur

    Parameters:
        df : Raw or partially preprocessed DataFrame (before scaling)

    Returns:
        df : DataFrame with engineered feature columns appended
    """
    required = {"sbytes", "dbytes", "spkts", "dpkts", "dur"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"Feature engineering requires columns missing from dataset: {missing}")

    for name, fn in ENGINEERED_FEATURE_MAP.items():
        df[name] = fn(df)

    return df

"""
src/association/apriori.py
--------------------------
Association Rule Mining via the Apriori algorithm (implemented from scratch).

Purpose:
    Discover co-occurring behavioral patterns in UNSW-NB15 network traffic.
    Rules are purely unsupervised — label columns are never used.

    Example mined rule:
        "protocol_tcp AND service_ftp → state_CON"
        support=0.032, confidence=0.81, lift=2.47

Pipeline stages (as defined in the assignment slides):
    STEP 1  — Market Basket construction
              Convert continuous features into 3-bin categorical strings,
              combine with native categorical features, build transaction sets.

    STEP 2  — Apriori frequent itemset mining
              Count 1-itemsets → generate & prune k-itemsets using the
              Apriori anti-monotone principle until no new itemsets exist.

    STEP 3  — Association rule generation
              For every frequent itemset of size ≥ 2 enumerate all
              antecedent/consequent splits and compute support, confidence,
              and lift.  Keep only rules where:
                  confidence ≥ min_confidence  AND  lift > 1

Public API:
    run_apriori(df, min_support, min_confidence) → dict
    save_rules_to_csv(rules, save_path)

Constraints honoured:
    • No mlxtend / external mining libraries used
    • No FP-Growth
    • No label information (attack_cat / label columns are dropped implicitly
      because they are never in the feature list)
    • Purely unsupervised
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Dict, List, Set

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# Native categorical columns present in UNSW-NB15 (raw string values kept as-is)
CATEGORICAL_FEATURES: List[str] = ["proto", "service", "state"]

# Continuous engineered / raw features to discretise into low/medium/high bins
CONTINUOUS_FEATURES: List[str] = [
    "outbound_dominance_ratio",
    "packet_rate",
    "bytes_per_packet",
    "packet_asymmetry",
]

# Number of quantile bins for continuous features
N_BINS: int = 3
BIN_LABELS: List[str] = ["low", "medium", "high"]


# ── STEP 1 — Market Basket Construction ───────────────────────────────────────

def _build_transactions(df: pd.DataFrame) -> List[Set[str]]:
    """
    Convert a raw/pre-engineered (not scaled) DataFrame into a list of
    transaction sets suitable for Apriori mining.

    Each transaction is a set of item strings, e.g.::

        {"proto_tcp", "service_ftp", "state_CON",
         "outbound_dominance_ratio_high", "packet_rate_low",
         "bytes_per_packet_medium", "packet_asymmetry_low"}

    Processing steps
    ----------------
    1. Retain only the features listed in CATEGORICAL_FEATURES and
       CONTINUOUS_FEATURES that actually exist in ``df``.
    2. For categorical columns: stringify the raw value and prefix with the
       column name, e.g. ``proto`` = ``tcp``  →  ``proto_tcp``.
    3. For continuous columns: apply quantile-based binning into 3 equal-
       frequency bins labelled low / medium / high,
       then prefix with the column name.
    4. Build one transaction set per row from all item strings.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-engineered DataFrame **before** scaling.  Must still contain
        the raw categorical columns (proto, service, state) and the
        engineered continuous columns.

    Returns
    -------
    List[Set[str]]
        One set per network session.
    """
    work = df.copy()

    item_columns: Dict[str, pd.Series] = {}

    # ── Categorical features ──────────────────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        if col not in work.columns:
            continue
        item_col = work[col].astype(str).str.strip().str.lower()
        item_col = col + "_" + item_col
        item_columns[col] = item_col

    # ── Continuous features → quantile bins ───────────────────────────────────
    for col in CONTINUOUS_FEATURES:
        if col not in work.columns:
            continue
        series = work[col].astype(float)

        # Compute quantile edges (33rd and 67th percentiles as bin boundaries)
        q33 = series.quantile(1 / N_BINS)
        q67 = series.quantile(2 / N_BINS)

        def _bin(val: float, lo=q33, hi=q67) -> str:
            if val <= lo:
                return "low"
            elif val <= hi:
                return "medium"
            else:
                return "high"

        binned = series.map(_bin)
        item_columns[col] = col + "_" + binned

    if not item_columns:
        raise ValueError(
            "No recognised features found in the DataFrame. "
            f"Expected at least one of: {CATEGORICAL_FEATURES + CONTINUOUS_FEATURES}"
        )

    # ── Assemble transactions ─────────────────────────────────────────────────
    item_df = pd.DataFrame(item_columns, index=work.index)
    transactions: List[Set[str]] = [
        set(row.dropna().values) for _, row in item_df.iterrows()
    ]
    return transactions


# ── STEP 2 — Apriori Frequent Itemset Mining ──────────────────────────────────

def _count_support(
    transactions: List[Set[str]],
    itemsets: List[frozenset],
) -> Dict[frozenset, float]:
    """
    Compute support for a list of candidate itemsets.

    Support formula::

        support(X) = count(transactions containing X) / total_transactions
    """
    total = len(transactions)
    support_map: Dict[frozenset, float] = {}
    for itemset in itemsets:
        count = sum(1 for txn in transactions if itemset.issubset(txn))
        support_map[itemset] = count / total
    return support_map


def _generate_candidates(
    frequent_itemsets: List[frozenset],
    k: int,
) -> List[frozenset]:
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets by
    joining pairs that share all but one item (F_{k-1} × F_{k-1} join).

    Apriori pruning: a candidate is retained only when ALL of its
    (k-1)-subsets appear in ``frequent_itemsets``.
    """
    frequent_set = set(frequent_itemsets)
    candidates: List[frozenset] = []
    seen: set = set()

    frequent_list = sorted([sorted(fs) for fs in frequent_itemsets])

    for i in range(len(frequent_list)):
        for j in range(i + 1, len(frequent_list)):
            a, b = frequent_list[i], frequent_list[j]
            if a[: k - 2] == b[: k - 2]:
                candidate = frozenset(a) | frozenset(b)
                if len(candidate) == k and candidate not in seen:
                    all_subsets_frequent = all(
                        frozenset(subset) in frequent_set
                        for subset in itertools.combinations(candidate, k - 1)
                    )
                    if all_subsets_frequent:
                        candidates.append(candidate)
                        seen.add(candidate)

    return candidates


def _apriori_frequent_itemsets(
    transactions: List[Set[str]],
    min_support: float,
) -> Dict[frozenset, float]:
    """
    Run the full Apriori frequent-itemset mining loop.

    Algorithm:
        1. Count all 1-itemsets; keep those with support ≥ min_support.
        2. Generate candidate 2-itemsets; prune; count; keep frequent.
        3. Repeat for k = 3, 4, … until no new frequent itemsets are found.
    """
    all_frequent: Dict[frozenset, float] = {}

    # ── Level 1: 1-itemsets ───────────────────────────────────────────────────
    unique_items: set = set()
    for txn in transactions:
        unique_items.update(txn)

    candidates_1 = [frozenset([item]) for item in unique_items]
    support_1 = _count_support(transactions, candidates_1)
    frequent_k = {
        fs: sup for fs, sup in support_1.items() if sup >= min_support
    }

    if not frequent_k:
        return all_frequent

    all_frequent.update(frequent_k)

    # ── Levels 2, 3, … ───────────────────────────────────────────────────────
    k = 2
    while frequent_k:
        candidates_k = _generate_candidates(list(frequent_k.keys()), k)
        if not candidates_k:
            break

        support_k = _count_support(transactions, candidates_k)
        frequent_k = {
            fs: sup for fs, sup in support_k.items() if sup >= min_support
        }

        all_frequent.update(frequent_k)
        k += 1

    return all_frequent


# ── STEP 3 — Association Rule Generation ──────────────────────────────────────

def _generate_rules(
    frequent_itemsets: Dict[frozenset, float],
    min_confidence: float,
) -> List[Dict]:
    """
    Generate association rules from all frequent itemsets of size ≥ 2.

    For every itemset X of size n ≥ 2:
        For every non-empty proper subset A ⊂ X:
            B = X \\ A
            Compute:
                support    = support(X)
                confidence = support(X) / support(A)
                lift       = confidence / support(B)

    Filtering: Keep rule only when confidence ≥ min_confidence AND lift > 1.
    """
    rules: List[Dict] = []

    for itemset, sup_itemset in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        items = list(itemset)

        for r in range(1, len(items)):
            for antecedent_tuple in itertools.combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent

                sup_antecedent = frequent_itemsets.get(antecedent)
                sup_consequent = frequent_itemsets.get(consequent)

                if sup_antecedent is None or sup_consequent is None:
                    continue
                if sup_antecedent == 0 or sup_consequent == 0:
                    continue

                confidence = sup_itemset / sup_antecedent
                lift = confidence / sup_consequent

                if confidence >= min_confidence and lift > 1.0:
                    rules.append({
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support":    round(sup_itemset, 6),
                        "confidence": round(confidence,  6),
                        "lift":       round(lift,        6),
                    })

    return rules


# ── Sanity Checks ─────────────────────────────────────────────────────────────

_BANNED_COLUMNS = {"label", "attack_cat"}


def _run_sanity_checks(rules: List[Dict]) -> None:
    """
    Post-generation sanity checks on the mined rule set.

    Checks: support in [0,1], confidence in [0,1], no NaN, no banned label
    tokens, lift > 1 for all rules.
    """
    separator = "-" * 55
    print()
    print(separator)
    print("  SANITY CHECKS")
    print(separator)

    if not rules:
        print("  [INFO] No rules to check — skipping sanity checks.")
        print(separator)
        return

    supports = [r["support"] for r in rules]
    assert all(not math.isnan(s) for s in supports), \
        "FAIL: NaN detected in support values!"
    assert all(0.0 <= s <= 1.0 for s in supports), \
        "FAIL: Support value out of [0, 1] range!"
    print("  [PASS] All support values are in [0, 1] — no NaN.")

    confidences = [r["confidence"] for r in rules]
    assert all(not math.isnan(c) for c in confidences), \
        "FAIL: NaN detected in confidence values!"
    assert all(0.0 <= c <= 1.0 for c in confidences), \
        "FAIL: Confidence value out of [0, 1] range!"
    print("  [PASS] All confidence values are in [0, 1] — no NaN.")

    for rule in rules:
        all_items = rule["antecedent"] | rule["consequent"]
        for item in all_items:
            token = item.split("_")[0]
            assert token not in _BANNED_COLUMNS, \
                f"FAIL: Banned column '{token}' found in rule item '{item}'!"
    print("  [PASS] No label / attack_cat tokens found in any rule.")

    lifts = [r["lift"] for r in rules]
    assert all(not math.isnan(lv) for lv in lifts), \
        "FAIL: NaN detected in lift values!"
    assert all(lv > 1.0 for lv in lifts), \
        "FAIL: Rule with lift ≤ 1 found (filter not applied)!"
    print(f"  [INFO] Lift range  : min={min(lifts):.4f}  max={max(lifts):.4f}")
    print("  [PASS] All lift values > 1.")

    print(separator)
    print()


# ── CSV Export ────────────────────────────────────────────────────────────────

def save_rules_to_csv(rules: List[Dict], save_path: str) -> None:
    """
    Persist association rules to a CSV file.

    The antecedent and consequent frozensets are serialised as
    human-readable strings: "item_a AND item_b".

    Columns: antecedent, consequent, support, confidence, lift
    """
    if not rules:
        print("  [WARN] No rules to save — CSV not written.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    rows = []
    for rule in rules:
        rows.append({
            "antecedent": " AND ".join(sorted(rule["antecedent"])),
            "consequent": " AND ".join(sorted(rule["consequent"])),
            "support":    rule["support"],
            "confidence": rule["confidence"],
            "lift":       rule["lift"],
        })

    df_out = pd.DataFrame(rows, columns=[
        "antecedent", "consequent", "support", "confidence", "lift"
    ])
    df_out = df_out.sort_values("lift", ascending=False).reset_index(drop=True)
    df_out.to_csv(save_path, index=False)
    print(f"  [SAVE] Association rules saved to: {save_path}")
    print(f"         Rows written: {len(df_out):,}")
    print()


# ── Terminal Printing ─────────────────────────────────────────────────────────

def _print_top_rules(rules: List[Dict], top_n: int = 20) -> None:
    """Print the top-N association rules sorted by lift (descending)."""
    sorted_rules = sorted(rules, key=lambda r: r["lift"], reverse=True)
    top = sorted_rules[:top_n]

    separator = "=" * 100
    print()
    print(separator)
    print(f"  TOP {top_n} ASSOCIATION RULES — sorted by Lift (descending)")
    print(separator)

    if not top:
        print("  No rules found satisfying the given thresholds.")
        print(separator)
        return

    header = (
        f"  {'Rank':>4}  "
        f"{'Antecedent':<45}  "
        f"{'Consequent':<30}  "
        f"{'Support':>8}  "
        f"{'Conf':>7}  "
        f"{'Lift':>7}"
    )
    print(header)
    print("-" * 100)

    for rank, rule in enumerate(top, start=1):
        ant_str = " AND ".join(sorted(rule["antecedent"]))
        con_str = " AND ".join(sorted(rule["consequent"]))
        if len(ant_str) > 43:
            ant_str = ant_str[:40] + "..."
        if len(con_str) > 28:
            con_str = con_str[:25] + "..."
        print(
            f"  {rank:>4}  "
            f"{ant_str:<45}  "
            f"{con_str:<30}  "
            f"{rule['support']:>8.4f}  "
            f"{rule['confidence']:>7.4f}  "
            f"{rule['lift']:>7.4f}"
        )

    print(separator)
    print(f"  Total rules mined (above thresholds): {len(rules)}")
    print(separator)
    print()


# ── Public API ────────────────────────────────────────────────────────────────

def run_apriori(
    df: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.7,
) -> dict:
    """
    Full Association Rule Mining pipeline using the Apriori algorithm.

    Accepts the **pre-engineered, unscaled** DataFrame (i.e. the DataFrame
    *after* ``apply_feature_engineering`` but *before* ``scale_features``).
    Label columns are not expected and will be silently ignored if present.

    Parameters
    ----------
    df : pd.DataFrame
        Raw/feature-engineered DataFrame (not scaled).
        Must contain at least some of:
            Categorical : proto, service, state
            Continuous  : outbound_dominance_ratio, packet_rate,
                          bytes_per_packet, packet_asymmetry
    min_support : float, optional
        Minimum support threshold.  Default = 0.01 (1% of transactions).
    min_confidence : float, optional
        Minimum confidence threshold.  Default = 0.70 (70%).

    Returns
    -------
    dict with keys:
        "frequent_itemsets" : dict mapping frozenset → float (support)
        "rules"             : list of dicts, each containing:
                                  antecedent  (frozenset)
                                  consequent  (frozenset)
                                  support     (float)
                                  confidence  (float)
                                  lift        (float)
    """
    print()
    print("=" * 55)
    print("ASSOCIATION RULE MINING — Apriori")
    print("=" * 55)
    print(f"  Dataset        : {len(df):,} sessions")
    print(f"  min_support    : {min_support}")
    print(f"  min_confidence : {min_confidence}")
    print()

    # ── STEP 1: Build transactions ────────────────────────────────────────────
    print("  [Step 1] Building Market Basket transactions …")
    transactions = _build_transactions(df)
    n_transactions = len(transactions)
    avg_items = np.mean([len(t) for t in transactions])
    print(f"           Transactions      : {n_transactions:,}")
    print(f"           Avg items/txn     : {avg_items:.2f}")
    print()

    # ── STEP 2: Mine frequent itemsets ────────────────────────────────────────
    print("  [Step 2] Mining frequent itemsets (Apriori) …")
    frequent_itemsets = _apriori_frequent_itemsets(transactions, min_support)
    size_dist: Dict[int, int] = {}
    for fs in frequent_itemsets:
        sz = len(fs)
        size_dist[sz] = size_dist.get(sz, 0) + 1

    print(f"           Frequent itemsets found: {len(frequent_itemsets):,}")
    for sz in sorted(size_dist):
        print(f"             Size {sz}: {size_dist[sz]:,} itemsets")
    print()

    # ── STEP 3: Generate association rules ───────────────────────────────────
    print("  [Step 3] Generating association rules …")
    rules = _generate_rules(frequent_itemsets, min_confidence)
    print(f"           Rules satisfying thresholds: {len(rules):,}")
    print()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    _run_sanity_checks(rules)

    # ── Print top rules ───────────────────────────────────────────────────────
    _print_top_rules(rules, top_n=20)

    return {
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
    }

"""
src/association/__init__.py
---------------------------
Association Rule Mining module — Apriori implemented from scratch.

Algorithm:
    Apriori frequent itemset mining on UNSW-NB15 discretised traffic features.
    Implemented without mlxtend or FP-Growth — purely numpy/pandas.

Use case:
    Discover co-occurring network behaviors:
    e.g. "proto_tcp AND packet_rate_high → outbound_dominance_ratio_high"
    Rules surface behavioral patterns linking protocol, service, state,
    and engineered features — interpretable for SOC analysts.

Integration point in main.py:
    from src.association.apriori import run_apriori, save_rules_to_csv
    apriori_result = run_apriori(df, min_support=0.01, min_confidence=0.70)
    save_rules_to_csv(apriori_result["rules"], save_path="outputs/association_rules.csv")
"""

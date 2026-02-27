"""
src/association/__init__.py
---------------------------
Association Rule Mining module — NOT YET IMPLEMENTED.

Planned algorithms:
    - Apriori        : Frequent itemset mining on discretised traffic features
    - FP-Growth      : Efficient frequent pattern discovery
    - Rule filtering : Support, confidence, lift threshold control

Use case:
    Discover co-occurring network behaviors:
    e.g. "sessions with high sbytes AND FTP service often have state=CON"
    Rules can inform signature-based detection systems.

Integration point in main.py:
    from src.association.apriori import run_apriori
    rules = run_apriori(df_discretised, min_support=0.01, min_confidence=0.8)
"""

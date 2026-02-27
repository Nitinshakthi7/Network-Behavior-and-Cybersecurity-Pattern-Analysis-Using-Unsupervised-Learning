"""
src/recommendation/__init__.py
-------------------------------
Recommender System module — NOT YET IMPLEMENTED.

Planned functionality:
    - Firewall rule suggestion based on cluster behavioral profiles
    - Risk scoring per session relative to cluster centroids
    - Automated policy recommendations for anomalous clusters

Use case:
    Given a session's cluster assignment and distance from centroid,
    recommend a security policy action (allow / inspect / block).

Integration point in main.py:
    from src.recommendation.policy import suggest_rules
    recommendations = suggest_rules(km_result, db_result, risk_threshold=0.8)
"""

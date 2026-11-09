"""
Central configuration for Model 5 (Graph Interaction Model).

LEAKAGE: `is_fraud`, `is_fraud_edge`, `ring_id`, and `ring_role` are never
read as model features anywhere in this package — they're labels or
generation scaffolding. `is_fraud_edge`/topology live only in
`output/edges_audit.csv`, which this package never loads for feature
construction (only for optional auditing).
"""

from __future__ import annotations

LEAKAGE_COLUMNS = frozenset({"is_fraud", "is_fraud_edge", "ring_id", "ring_role"})

# "New account" proxy for the new-account-contact-share feature and the
# isolation-pattern signal: a contact is "new" if the edge happened within
# this many days of the contact's own account creation.
NEW_ACCOUNT_WINDOW_DAYS = 14

# A node's own activity is "bursty" if this fraction of its outgoing
# interactions land in the first 48h after its own account creation.
BURST_WINDOW_HOURS = 48

# Degree-only baseline (Step 3): precision is reported at these fixed
# recall operating points, per fraud type — not just AUC. With a large
# legit low-activity tail, "few connections" alone will flag a lot of
# innocent users; precision at a recall point shows that cost directly.
PRECISION_AT_RECALL_TARGETS = (0.50, 0.80)

RANDOM_SEED = 42

"""Node feature engineering for Model 5.

Every feature here is derived purely from graph structure (edges.csv) and
non-label profile attributes (created_at, age, city). None of them ever
touch `is_fraud`, `is_fraud_edge`, `ring_id`, or `ring_role` — see
config.LEAKAGE_COLUMNS.

NOTE: profiles.csv also has `messages_sent`/`messages_received` columns,
but those are an independent self-reported activity proxy generated before
the graph exists (see dataset/legitimate_generator.py), not reconciled with
edges.csv (correlation ~0.55/0.07). This module computes degree/message
counts from edges.csv directly instead — the two are deliberately different
quantities, not duplicates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model5.graph_interaction.config import BURST_WINDOW_HOURS, NEW_ACCOUNT_WINDOW_DAYS

RAW_FEATURE_COLUMNS = [
    "in_degree",
    "out_degree",
    "degree_ratio",
    "response_rate",
    "unique_contact_ratio",
    "burst_score",
    "new_account_contact_share",
    "account_age_at_first_interaction_days",
    "contact_age_std",
    "contact_city_diversity",
]


def _reciprocal_flags(pairs: pd.DataFrame) -> pd.Series:
    edge_set = set(zip(pairs["source"], pairs["target"]))
    return pd.Series(
        [(t, s) in edge_set for s, t in zip(pairs["source"], pairs["target"])],
        index=pairs.index,
    )


def build_node_features(profiles_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per profile in profiles_df: profile_id, raw feature
    columns (RAW_FEATURE_COLUMNS), and metadata columns (is_fraud, fraud_type,
    ring_role) kept separate for evaluation — never fed to a model as input.
    """
    all_ids = profiles_df[["profile_id"]].copy()
    created_lookup = profiles_df.set_index("profile_id")["created_at_dt"]
    age_lookup = profiles_df.set_index("profile_id")["age"]
    city_lookup = profiles_df.set_index("profile_id")["city"]

    out_degree = edges_df.groupby("source").size().rename("out_degree")
    in_degree = edges_df.groupby("target").size().rename("in_degree")

    # Response rate: fraction of a node's distinct outgoing contacts that it
    # also has an edge back FROM (any edge, not relying on a "reply" label —
    # that distinction only exists in edges_audit.csv, which is off-limits).
    pairs = edges_df[["source", "target"]].drop_duplicates()
    pairs["has_reciprocal"] = _reciprocal_flags(pairs)
    recip_counts = pairs[pairs["has_reciprocal"]].groupby("source").size().rename("recip_count")
    unique_out_pairs = pairs.groupby("source").size().rename("unique_out_pairs")

    unique_out_neighbors = edges_df.groupby("source")["target"].nunique().rename("unique_out_neighbors")

    # Burst score: share of a node's own outgoing edges inside the first
    # BURST_WINDOW_HOURS after its own account creation.
    out_edges = edges_df[["source", "target", "timestamp_dt"]].copy()
    out_edges["source_created"] = out_edges["source"].map(created_lookup)
    out_edges["hours_since_creation"] = (out_edges["timestamp_dt"] - out_edges["source_created"]).dt.total_seconds() / 3600.0
    out_edges["is_burst"] = out_edges["hours_since_creation"] <= BURST_WINDOW_HOURS
    burst_score = out_edges.groupby("source")["is_burst"].mean().rename("burst_score")

    # New-account contact share: of a node's outgoing contacts, what
    # fraction were themselves very recently created at the time of contact.
    out_edges["target_created"] = out_edges["target"].map(created_lookup)
    out_edges["target_age_days_at_contact"] = (out_edges["timestamp_dt"] - out_edges["target_created"]).dt.total_seconds() / 86400.0
    out_edges["target_is_new"] = out_edges["target_age_days_at_contact"] <= NEW_ACCOUNT_WINDOW_DAYS
    new_account_share = out_edges.groupby("source")["target_is_new"].mean().rename("new_account_contact_share")

    # Demographic spread of outgoing contacts (age std, city diversity).
    out_edges["target_age"] = out_edges["target"].map(age_lookup)
    out_edges["target_city"] = out_edges["target"].map(city_lookup)
    contact_age_std = out_edges.groupby("source")["target_age"].std(ddof=0).rename("contact_age_std")
    contact_city_diversity = (
        out_edges.groupby("source")["target_city"].nunique() / out_edges.groupby("source").size()
    ).rename("contact_city_diversity")

    # Account age at first interaction: earliest edge touching this node
    # (either direction) minus its own created_at.
    touches = pd.concat([
        edges_df[["source", "timestamp_dt"]].rename(columns={"source": "profile_id"}),
        edges_df[["target", "timestamp_dt"]].rename(columns={"target": "profile_id"}),
    ])
    first_touch = touches.groupby("profile_id")["timestamp_dt"].min().rename("first_touch")

    feat = all_ids.set_index("profile_id")
    feat = feat.join(in_degree).join(out_degree)
    feat = feat.join(recip_counts).join(unique_out_pairs).join(unique_out_neighbors)
    feat = feat.join(burst_score).join(new_account_share)
    feat = feat.join(contact_age_std).join(contact_city_diversity)
    feat = feat.join(first_touch)
    feat["own_created"] = feat.index.map(created_lookup)

    feat["in_degree"] = feat["in_degree"].fillna(0)
    feat["out_degree"] = feat["out_degree"].fillna(0)
    feat["degree_ratio"] = feat["out_degree"] / feat["in_degree"].clip(lower=1)
    feat["response_rate"] = (feat["recip_count"].fillna(0) / feat["unique_out_pairs"].clip(lower=1)).where(feat["out_degree"] > 0, 0.0)
    feat["unique_contact_ratio"] = (feat["unique_out_neighbors"].fillna(0) / feat["out_degree"].clip(lower=1)).where(feat["out_degree"] > 0, 0.0)
    feat["burst_score"] = feat["burst_score"].fillna(0.0)
    feat["new_account_contact_share"] = feat["new_account_contact_share"].fillna(0.0)
    feat["contact_age_std"] = feat["contact_age_std"].fillna(0.0)
    feat["contact_city_diversity"] = feat["contact_city_diversity"].fillna(0.0)
    feat["account_age_at_first_interaction_days"] = (
        (feat["first_touch"] - feat["own_created"]).dt.total_seconds() / 86400.0
    ).fillna(-1.0)  # -1 sentinel: never touched by any edge

    out = feat.reset_index()[["profile_id"] + RAW_FEATURE_COLUMNS]
    out = out.merge(
        profiles_df[["profile_id", "is_fraud", "fraud_type", "ring_role"]],
        on="profile_id", how="left",
    )
    return out


def normalize_features(features_df: pd.DataFrame, train_profile_ids: set) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on TRAIN rows only, transform every row."""
    train_mask = features_df["profile_id"].isin(train_profile_ids)
    X = features_df[RAW_FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    scaler = StandardScaler().fit(X[train_mask.to_numpy()])
    scaled = pd.DataFrame(scaler.transform(X), columns=RAW_FEATURE_COLUMNS, index=features_df.index)
    scaled.insert(0, "profile_id", features_df["profile_id"].to_numpy())
    return scaled, scaler

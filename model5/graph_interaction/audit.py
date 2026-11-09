"""Data loading, validation, and audit reporting for Model 5."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_PROFILE_COLUMNS = {"profile_id", "created_at", "is_fraud", "fraud_type"}
REQUIRED_EDGE_COLUMNS = {"source", "target", "message_count", "match_requested", "timestamp"}


@dataclass
class GraphAuditResult:
    text: str
    warnings: list[str]


def load_and_validate_profiles(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_PROFILE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    df["created_at_dt"] = pd.to_datetime(df["created_at"], format="ISO8601", errors="coerce")
    return df


def load_and_validate_edges(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_EDGE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    if "is_fraud_edge" in df.columns:
        # edges.csv should never carry this; if it does (e.g. someone pointed
        # this loader at edges_audit.csv by mistake), fail loudly rather than
        # silently risk it being treated as a feature downstream.
        raise ValueError(
            f"{csv_path} contains 'is_fraud_edge' — this looks like the audit file, "
            "not the model-facing edges.csv. Refusing to load it as graph input."
        )
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    return df


def _causality_violations(profiles_df: pd.DataFrame, edges_df: pd.DataFrame) -> int:
    created_lookup = profiles_df.set_index("profile_id")["created_at_dt"]
    src_created = edges_df["source"].map(created_lookup)
    tgt_created = edges_df["target"].map(created_lookup)
    latest_endpoint = pd.concat([src_created, tgt_created], axis=1).max(axis=1)
    return int((edges_df["timestamp_dt"] < latest_endpoint).sum())


def generate_graph_audit(profiles_df: pd.DataFrame, edges_df: pd.DataFrame) -> GraphAuditResult:
    all_ids = set(profiles_df["profile_id"])
    touched_ids = set(edges_df["source"]) | set(edges_df["target"])

    out_deg = edges_df.groupby("source").size()
    in_deg = edges_df.groupby("target").size()
    degree = pd.DataFrame({"profile_id": list(all_ids)})
    degree["out_degree"] = degree["profile_id"].map(out_deg).fillna(0).astype(int)
    degree["in_degree"] = degree["profile_id"].map(in_deg).fillna(0).astype(int)
    degree["total_degree"] = degree["out_degree"] + degree["in_degree"]
    degree = degree.merge(profiles_df[["profile_id", "fraud_type", "is_fraud", "ring_role"]], on="profile_id", how="left")
    degree["group"] = degree["fraud_type"].fillna("legitimate")

    coverage_rows = []
    for group, g in degree.groupby("group"):
        coverage_rows.append({
            "group": group,
            "count": len(g),
            "zero_degree": int((g["total_degree"] == 0).sum()),
            "mean_degree": float(g["total_degree"].mean()),
            "median_degree": float(g["total_degree"].median()),
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values("count", ascending=False)

    sat = degree[degree["ring_role"] == "satellite"]
    relay = degree[degree["ring_role"] == "relay"]
    hub = degree[degree["ring_role"] == "hub"]

    violations = _causality_violations(profiles_df, edges_df)
    orphan_edges = int((~edges_df["source"].isin(all_ids) | ~edges_df["target"].isin(all_ids)).sum())

    lines = [
        "=" * 78,
        "MODEL 5 GRAPH AUDIT",
        "=" * 78,
        f"Total profiles: {len(profiles_df):,}",
        f"Total edges: {len(edges_df):,}",
        f"Profiles touched by >=1 edge: {len(touched_ids):,} / {len(all_ids):,}",
        f"Orphan edges (endpoint not in profiles.csv): {orphan_edges:,}",
        f"Timestamp-causality violations (edge before both endpoints existed): {violations:,}",
        "",
        "Degree by group:",
        coverage_df.to_string(index=False),
        "",
        "Ring-role coverage (should all be 0 zero-degree after the Step 0 fix):",
        f"  satellites: {len(sat):,} profiles, {int((sat['total_degree']==0).sum())} zero-degree, mean degree {sat['total_degree'].mean():.2f}",
        f"  relays:     {len(relay):,} profiles, {int((relay['total_degree']==0).sum())} zero-degree, mean degree {relay['total_degree'].mean():.2f}",
        f"  hubs:       {len(hub):,} profiles, {int((hub['total_degree']==0).sum())} zero-degree, mean degree {hub['total_degree'].mean():.2f}",
    ]

    warnings = []
    if violations:
        warnings.append(f"{violations} edges have a timestamp before one of their endpoints' created_at.")
    if orphan_edges:
        warnings.append(f"{orphan_edges} edges reference a profile_id not present in profiles.csv.")
    zero_fraud = coverage_df[(coverage_df["group"] != "legitimate") & (coverage_df["zero_degree"] > 0)]
    if not zero_fraud.empty:
        warnings.append(f"Fraud groups with >0 zero-degree nodes: {zero_fraud['group'].tolist()}")

    return GraphAuditResult("\n".join(lines), warnings)


def save_audit(profiles_df: pd.DataFrame, edges_df: pd.DataFrame, output_dir: str) -> GraphAuditResult:
    os.makedirs(output_dir, exist_ok=True)
    result = generate_graph_audit(profiles_df, edges_df)
    with open(os.path.join(output_dir, "graph_audit.txt"), "w", encoding="utf-8") as f:
        f.write(result.text)
    return result

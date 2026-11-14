"""
regenerate_edges.py

Rebuilds output/edges.csv, output/graph_stats.csv, and output/edges_audit.csv
from an already-generated profiles.csv, without touching profiles.csv,
profiles.parquet, the splits, or reports.csv. Use this after changing
graph_generator.py's logic so you don't have to re-run the full
50,000-profile generation just to get new graph data.

Run from the project root:
    python -m dataset.regenerate_edges --profiles ./output/profiles.csv --output_dir ./output
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

if __package__:
    from .graph_generator import build_interaction_graph
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from dataset.graph_generator import build_interaction_graph

REQUIRED_COLUMNS = [
    "profile_id", "is_fraud", "fraud_type", "ring_id", "ring_role",
    "unique_contacts", "age", "gender", "city", "state",
    "annual_income_lpa", "created_at",
]


def main():
    parser = argparse.ArgumentParser(description="Regenerate output/edges.csv from an existing profiles.csv")
    parser.add_argument("--profiles", default="./output/profiles.csv")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading profiles from {args.profiles}...")
    df = pd.read_csv(args.profiles)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"profiles.csv is missing columns needed for graph generation: {missing}")
    print(f"  {len(df):,} profiles loaded")

    print("Building interaction graph...")
    G, edges_df, stats_df, audit_df = build_interaction_graph(df, seed=args.seed)

    edges_path = os.path.join(args.output_dir, "edges.csv")
    stats_path = os.path.join(args.output_dir, "graph_stats.csv")
    audit_path = os.path.join(args.output_dir, "edges_audit.csv")
    edges_df.to_csv(edges_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"Saved: {edges_path}")
    print(f"Saved: {stats_path}")
    print(f"Saved: {audit_path} (audit only — is_fraud_edge/topology, never a model input)")


if __name__ == "__main__":
    main()

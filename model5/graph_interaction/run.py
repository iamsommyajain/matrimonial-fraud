"""Main runner for Model 5, Steps 1-3: audit, features, graph sanity-check,
and the degree-only baseline. Does NOT train a GAT — that's Steps 4-6.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model5.graph_interaction.audit import load_and_validate_edges, load_and_validate_profiles, save_audit
from model5.graph_interaction.config import PRECISION_AT_RECALL_TARGETS, RANDOM_SEED
from model5.graph_interaction.evaluate import build_baseline_report, evaluate_group_vs_legit
from model5.graph_interaction.features import build_node_features, normalize_features
from model5.graph_interaction.graph_builder import build_pyg_data

DEGREE_FEATURES = ["in_degree", "out_degree", "degree_ratio"]


def _degree_only_matrix(features_df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.log1p(features_df["in_degree"].to_numpy(dtype=float)),
        np.log1p(features_df["out_degree"].to_numpy(dtype=float)),
        np.log1p(features_df["degree_ratio"].to_numpy(dtype=float)),
    ])


def _fit_degree_model(train_df: pd.DataFrame, seed: int) -> tuple[LogisticRegression, StandardScaler]:
    X = _degree_only_matrix(train_df)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(X), train_df["is_fraud"].astype(int).to_numpy())
    return clf, scaler


def _score(clf: LogisticRegression, scaler: StandardScaler, df: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(scaler.transform(_degree_only_matrix(df)))[:, 1]


def main():
    parser = argparse.ArgumentParser(description="Model 5 Steps 1-3: audit, features, degree-only baseline")
    parser.add_argument("--profiles", default="./output/profiles.csv")
    parser.add_argument("--edges", default="./output/edges.csv")
    parser.add_argument("--output_dir", default="./model5/outputs")
    parser.add_argument("--splits_dir", default=None, help="defaults to <profiles_dir>/splits")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    splits_dir = args.splits_dir or os.path.join(os.path.dirname(args.profiles), "splits")

    print("Loading and validating profiles + edges...", flush=True)
    profiles_df = load_and_validate_profiles(args.profiles)
    edges_df = load_and_validate_edges(args.edges)

    print("Running graph audit...", flush=True)
    audit = save_audit(profiles_df, edges_df, args.output_dir)
    print(audit.text, flush=True)
    for w in audit.warnings:
        print(f"WARNING: {w}", flush=True)

    print("\nBuilding node features...", flush=True)
    features_df = build_node_features(profiles_df, edges_df)
    features_df.to_csv(os.path.join(args.output_dir, "m5_node_features.csv"), index=False)
    print(f"Saved node features ({len(features_df):,} rows) to {args.output_dir}/m5_node_features.csv", flush=True)

    train_ids = set(pd.read_csv(os.path.join(splits_dir, "train.csv"))["profile_id"])
    val_ids = set(pd.read_csv(os.path.join(splits_dir, "val.csv"))["profile_id"])
    test_ids = set(pd.read_csv(os.path.join(splits_dir, "test.csv"))["profile_id"])

    scaled_features_df, scaler = normalize_features(features_df, train_ids)
    scaled_features_df.to_csv(os.path.join(args.output_dir, "m5_node_features_normalized.csv"), index=False)

    print("\nBuilding PyG Data object (sanity check only — no training)...", flush=True)
    build = build_pyg_data(profiles_df, edges_df, scaled_features_df, train_ids, val_ids, test_ids)
    data = build.data
    print(
        f"  Data(x={list(data.x.shape)}, edge_index={list(data.edge_index.shape)}, "
        f"y={list(data.y.shape)}, train={int(data.train_mask.sum())}, "
        f"val={int(data.val_mask.sum())}, test={int(data.test_mask.sum())})",
        flush=True,
    )

    # ── Step 3: degree-only baseline ────────────────────────────────────────
    print("\nFitting degree-only baseline (in_degree, out_degree, degree_ratio)...", flush=True)
    train_df = features_df[features_df["profile_id"].isin(train_ids)]
    test_df = features_df[features_df["profile_id"].isin(test_ids)]

    clf, scaler_deg = _fit_degree_model(train_df, args.seed)
    test_df = test_df.copy()
    test_df["score"] = _score(clf, scaler_deg, test_df)
    legit_test_scores = test_df[test_df["fraud_type"].isna()]["score"].to_numpy()

    rows = []
    for ft in ["functional", "template_bio", "financial_scam", "multi", "coordinated_ring"]:
        sub = test_df[test_df["fraud_type"] == ft]
        if sub.empty:
            continue
        r = evaluate_group_vs_legit(sub["score"].to_numpy(), legit_test_scores, PRECISION_AT_RECALL_TARGETS)
        r["group"] = ft
        r["evaluation"] = "temporal_test (held-out)"
        rows.append(r)

    # coordinated_ring now genuinely appears in test (ring generation was
    # fixed to spread across all 6 months) — report satellites/hub+relay as
    # a true held-out breakdown of that same test-split ring population,
    # not a within-train proxy.
    for name, mask in [
        ("satellites only", (test_df["fraud_type"] == "coordinated_ring") & (test_df["ring_role"] == "satellite")),
        ("hub+relay only", (test_df["fraud_type"] == "coordinated_ring") & (test_df["ring_role"].isin(["hub", "relay"]))),
    ]:
        sub = test_df[mask]
        if sub.empty:
            continue
        r = evaluate_group_vs_legit(sub["score"].to_numpy(), legit_test_scores, PRECISION_AT_RECALL_TARGETS)
        r["group"] = name
        r["evaluation"] = "temporal_test (held-out)"
        rows.append(r)

    report = build_baseline_report(rows)
    print("\n" + report, flush=True)
    with open(os.path.join(args.output_dir, "m5_baseline_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "m5_baseline_metrics.csv"), index=False)
    print(f"\nSaved baseline report to {args.output_dir}/m5_baseline_report.txt", flush=True)
    print(f"Saved baseline metrics to {args.output_dir}/m5_baseline_metrics.csv", flush=True)


if __name__ == "__main__":
    main()

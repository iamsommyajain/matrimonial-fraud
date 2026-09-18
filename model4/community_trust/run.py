"""Main runner for Model 4 (Community Trust Model)."""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model4.community_trust.audit import load_and_validate_profiles, load_and_validate_reports, save_trust_audit
from model4.community_trust.aggregation import aggregate_trust_scores
from model4.community_trust.config import risk_level
from model4.community_trust.evaluate import build_concise_report, build_report


def _target_labels(df: pd.DataFrame) -> pd.Series:
    # Community trust reports are filed regardless of fraud sub-type, so the
    # target is simply "is this profile fraudulent", unlike M1/M2 which each
    # target one fraud sub-type.
    return df["is_fraud"].astype(bool).astype(int)


def _split_dir_from_profiles(profiles_path: str) -> str | None:
    base = os.path.dirname(profiles_path)
    split_dir = os.path.join(base, "splits")
    return split_dir if os.path.exists(split_dir) else None


def _select_threshold(val_scores: pd.DataFrame) -> float:
    y = val_scores["m4_target"].astype(int).to_numpy()
    s = val_scores["community_trust_risk"].to_numpy()
    rows = []
    for t in [i / 100 for i in range(101)]:
        pred = (s >= t).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((t, precision, recall, f1))
    return max(rows, key=lambda x: (x[3], x[2], x[1]))[0]


def main():
    parser = argparse.ArgumentParser(description="Run Model 4 community trust scoring")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting Model 4 run...", flush=True)
    print(f"Profiles   : {args.profiles}", flush=True)
    print(f"Reports    : {args.reports}", flush=True)
    print(f"Output dir : {args.output_dir}", flush=True)

    start = time.perf_counter()
    profiles_df = load_and_validate_profiles(args.profiles)
    reports_df = load_and_validate_reports(args.reports)
    profiles_df["m4_target"] = _target_labels(profiles_df)

    audit = save_trust_audit(profiles_df, reports_df, args.output_dir)
    print(audit.text[:2000], flush=True)
    for w in audit.warnings:
        print(f"WARNING: {w}", flush=True)

    agg_start = time.perf_counter()
    artifacts = aggregate_trust_scores(reports_df, profiles_df["profile_id"])
    agg_time = time.perf_counter() - agg_start

    scores_df = artifacts.scores.merge(
        profiles_df[["profile_id", "fraud_type", "is_fraud", "m4_target"]], on="profile_id", how="left"
    )
    scores_df["fraud_type"] = scores_df["fraud_type"].fillna("legitimate")
    scores_df["risk_level"] = scores_df["community_trust_risk"].map(risk_level)
    scores_df["aggregation_time_sec"] = agg_time
    scores_path = os.path.join(args.output_dir, "m4_scores.csv")
    scores_df.to_csv(scores_path, index=False)

    weighted_reports_path = os.path.join(args.output_dir, "m4_weighted_reports.csv")
    artifacts.weighted_reports.to_csv(weighted_reports_path, index=False)

    threshold = 0.5
    exploratory = True
    split_dir = _split_dir_from_profiles(args.profiles)
    evaluation_df = scores_df
    evaluation_scope = "full_dataset"
    if split_dir and os.path.exists(os.path.join(split_dir, "val.csv")) and os.path.exists(os.path.join(split_dir, "test.csv")):
        val_ids = set(pd.read_csv(os.path.join(split_dir, "val.csv"))["profile_id"])
        test_ids = set(pd.read_csv(os.path.join(split_dir, "test.csv"))["profile_id"])
        val_scores = scores_df[scores_df["profile_id"].isin(val_ids)]
        test_scores = scores_df[scores_df["profile_id"].isin(test_ids)]
        if not val_scores.empty and not test_scores.empty:
            threshold = _select_threshold(val_scores)
            evaluation_df = test_scores
            evaluation_scope = "temporal_test"
            exploratory = False

    runtime = {
        "aggregation_time_sec": agg_time,
        "total_time_sec": time.perf_counter() - start,
    }
    report = build_report(
        scores_df, threshold, args.output_dir, runtime,
        exploratory=exploratory, evaluation_df=evaluation_df, evaluation_scope=evaluation_scope,
    )
    print(build_concise_report(scores_df, threshold, evaluation_df=evaluation_df, evaluation_scope=evaluation_scope))
    print(f"Saved scores to {scores_path}")
    print(f"Saved weighted reports to {weighted_reports_path}")
    print(f"Saved audit to {os.path.join(args.output_dir, 'trust_audit.txt')}")
    print(f"Saved report to {os.path.join(args.output_dir, 'm4_report.txt')}")


if __name__ == "__main__":
    main()

"""Main runner for Model 2B."""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataset_generation.model2.behavior_audit import load_and_validate_profiles, save_behavior_audit
from dataset_generation.model2.behavior_anomaly import score_behavior
from dataset_generation.model2.behavior_features import build_behavior_features, preprocess_behavior_features
from dataset_generation.model2.evaluate_m2b import build_report


def _split_dir_from_profiles(profiles_path: str) -> str | None:
    base = os.path.dirname(profiles_path)
    split_dir = os.path.join(base, "splits")
    return split_dir if os.path.exists(split_dir) else None


def _load_split_profiles(split_dir: str):
    out = {}
    for name in ("train", "val", "test"):
        path = os.path.join(split_dir, f"{name}.csv")
        if os.path.exists(path):
            out[name] = pd.read_csv(path)
    return out


def _target_labels(df: pd.DataFrame) -> pd.Series:
    fraud_type = df["fraud_type"].fillna("")
    signals = df.get("injected_signals_str", pd.Series([""] * len(df), index=df.index)).fillna("")
    behavior_signals = [
        "burst_login_pattern",
        "rapid_photo_upload",
        "excessive_edits",
        "impossible_geo_logins",
        "behavioral_imbalance",
        "high_outreach_low_response",
        "sudden_inactivity_post_reports",
    ]
    mask = pd.Series(False, index=df.index)
    for sig in behavior_signals:
        mask |= signals.str.contains(sig, case=False, na=False)
    mask |= fraud_type.isin(["template_bio", "functional", "financial_scam"]) & mask
    mask |= fraud_type.eq("multi") & signals.str.contains("behavior", case=False, na=False)
    return mask.astype(int)


def main():
    parser = argparse.ArgumentParser(description="Run Model 2B behavioral anomaly detection")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--behavior", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--experiment", choices=["unsupervised", "novelty", "both"], default="both")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pd.options.mode.chained_assignment = None

    print("Starting Model 2B run...", flush=True)
    print(f"Profiles   : {args.profiles}", flush=True)
    print(f"Behavior   : {args.behavior or 'auto-detect from profiles'}", flush=True)
    print(f"Output dir : {args.output_dir}", flush=True)
    print(f"Experiment : {args.experiment}", flush=True)

    start = time.perf_counter()
    profiles_df = load_and_validate_profiles(args.profiles)
    profiles_df["m2b_target"] = _target_labels(profiles_df)
    audit = save_behavior_audit(profiles_df, args.output_dir)
    print(audit.text[:2000], flush=True)
    if audit.warnings:
        for w in audit.warnings:
            print(f"WARNING: {w}", flush=True)

    feat_start = time.perf_counter()
    features = build_behavior_features(profiles_df)
    feat_time = time.perf_counter() - feat_start
    features_path = os.path.join(args.output_dir, "m2b_features.csv")
    features.to_csv(features_path, index=False)

    prep_start = time.perf_counter()
    X, scaler, feature_columns = preprocess_behavior_features(features)
    prep_time = time.perf_counter() - prep_start

    split_dir = _split_dir_from_profiles(args.profiles)
    novelty_mask = None
    if split_dir and os.path.exists(os.path.join(split_dir, "train.csv")):
        splits = _load_split_profiles(split_dir)
        if "train" in splits:
            train_features = build_behavior_features(splits["train"])
            novelty_mask = pd.Series(features["profile_id"].isin(train_features["profile_id"]), index=features.index)

    score_start = time.perf_counter()
    scored = score_behavior(
        features,
        X,
        scaler,
        feature_columns,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        novelty_train_mask=(features["fraud_type"].fillna("").eq("legitimate") if novelty_mask is None else novelty_mask),
        experiment=args.experiment,
    )
    score_time = time.perf_counter() - score_start
    scores_df = scored.scores.copy()
    scores_df["behavior_aggregation_time_sec"] = feat_time
    scores_df["preprocess_time_sec"] = prep_time
    scores_df["scoring_time_sec"] = score_time
    scores_df["profiles_per_sec"] = len(scores_df) / max(score_time + feat_time + prep_time, 1e-9)
    scores_path = os.path.join(args.output_dir, "m2b_scores.csv")
    scores_df.to_csv(scores_path, index=False)

    threshold = 0.5
    exploratory = True
    if split_dir and os.path.exists(os.path.join(split_dir, "val.csv")):
        val = pd.read_csv(os.path.join(split_dir, "val.csv"))
        val_features = build_behavior_features(val)
        val_X, _, _ = preprocess_behavior_features(val_features)
        val_scored = score_behavior(
            val_features,
            val_X,
            scaler,
            feature_columns,
            n_estimators=args.n_estimators,
            random_state=args.seed,
            experiment=args.experiment,
        )
        val_df = val_scored.scores
        if not val_df.empty:
            y = val_df["m2b_target"].astype(int).to_numpy()
            s = val_df["behavior_risk"].to_numpy()
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
            threshold = max(rows, key=lambda x: (x[3], x[2], x[1]))[0]
            exploratory = False

    report = build_report(
        scores_df,
        threshold,
        args.output_dir,
        {
            "behavior_aggregation_time_sec": feat_time,
            "feature_preprocess_time_sec": prep_time,
            "scoring_time_sec": score_time,
            "total_time_sec": time.perf_counter() - start,
        },
        experiment=args.experiment,
        exploratory=exploratory,
    )
    print(report)
    print(f"Saved scores to {scores_path}")
    print(f"Saved features to {features_path}")
    print(f"Saved audit to {os.path.join(args.output_dir, 'behavior_audit.txt')}")


if __name__ == "__main__":
    main()

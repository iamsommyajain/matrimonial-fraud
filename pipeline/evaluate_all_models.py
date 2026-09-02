"""Unified temporal evaluation for M1, M2-A, M2-B, and score fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from dataset_generation.model1.evaluate_m1 import load_profiles, run_scoring


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def signal_mask(df: pd.DataFrame, names: list[str]) -> pd.Series:
    signals = df.get("injected_signals_str", pd.Series("", index=df.index)).fillna("")
    return signals.str.contains("|".join(names), case=False, na=False, regex=True)


def targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    fraud_type = df["fraud_type"].fillna("legitimate")
    functional_signal = signal_mask(df, ["age_experience", "education_profession", "income_company", "impossible_geo", "behavioral_imbalance"])
    text_signal = signal_mask(df, ["template_bio", "text_template", "bio_template"])
    behavior_signal = signal_mask(df, ["burst_login_pattern", "rapid_photo_upload", "excessive_edits", "impossible_geo_logins", "behavioral_imbalance", "high_outreach_low_response", "sudden_inactivity_post_reports"])
    return {
        "m1_primary": (fraud_type.eq("functional")),
        "m1_secondary": fraud_type.eq("functional") | (fraud_type.eq("multi") & functional_signal),
        "m2a_primary": fraud_type.eq("template_bio") | (fraud_type.eq("multi") & text_signal),
        "m2b_primary": behavior_signal & fraud_type.isin(["functional", "financial_scam", "template_bio", "multi"]),
        "all_fraud": df["is_fraud"].map(truthy),
    }


def best_threshold(y, score) -> float:
    rows = []
    for threshold in np.linspace(0.0, 1.0, 101):
        pred = score >= threshold
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((f1, recall, precision, threshold))
    return float(max(rows)[3])


def metric_row(model, scope, y, score, threshold, n_profiles):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ranked = np.argsort(-score)
    total = int(y.sum())
    row = {"model": model, "scope": scope, "profiles": n_profiles, "positives": total, "threshold": threshold}
    row["roc_auc"] = roc_auc_score(y, score) if len(np.unique(y)) > 1 else 0.5
    row["pr_auc"] = average_precision_score(y, score) if len(np.unique(y)) > 1 else 0.0
    row["precision"] = tp / (tp + fp) if tp + fp else 0.0
    row["recall"] = tp / (tp + fn) if tp + fn else 0.0
    row["f1"] = 2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"]) if row["precision"] + row["recall"] else 0.0
    row["fpr"] = fp / (fp + tn) if fp + tn else 0.0
    row["fnr"] = fn / (fn + tp) if fn + tp else 0.0
    for k in (100, 500, 1000):
        selected = ranked[:min(k, len(ranked))]
        row[f"precision_at_{k}"] = float(y[selected].mean()) if len(selected) else 0.0
        row[f"recall_at_{k}"] = float(y[selected].sum() / total) if total else 0.0
    for pct in (0.01, 0.02, 0.05, 0.10):
        selected = ranked[:max(1, int(len(ranked) * pct))]
        row[f"recall_at_{int(pct * 100)}pct"] = float(y[selected].sum() / total) if total else 0.0
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate all fraud models on temporal test data")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--m2a_scores", default="./model2/outputs/m2a_scores.csv")
    parser.add_argument("--m2b_scores", default="./model2/outputs/m2b_scores.csv")
    parser.add_argument("--output_dir", default="./pipeline/outputs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    profile_df = pd.read_csv(args.profiles)
    split_dir = os.path.join(os.path.dirname(args.profiles), "splits")
    train = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    target_map = targets(profile_df.set_index("profile_id").loc[profile_df["profile_id"]].reset_index())
    data = profile_df[["profile_id", "fraud_type", "is_fraud"]].copy()
    for name, target in target_map.items():
        data[name] = target.to_numpy()

    m1 = run_scoring(load_profiles(args.profiles))[["profile_id", "functional_risk_score"]]
    m2a = pd.read_csv(args.m2a_scores)[["profile_id", "text_risk"]]
    m2b = pd.read_csv(args.m2b_scores)[["profile_id", "behavior_risk"]]
    scores = data.merge(m1, on="profile_id").merge(m2a, on="profile_id").merge(m2b, on="profile_id")
    scores["fusion_risk"] = 1.0 - (1.0 - scores["functional_risk_score"]) * (1.0 - scores["text_risk"]) * (1.0 - scores["behavior_risk"])
    scores.to_csv(os.path.join(args.output_dir, "all_model_scores.csv"), index=False)

    val_ids, test_ids = set(val["profile_id"]), set(test["profile_id"])
    val_scores = scores[scores.profile_id.isin(val_ids)]
    test_scores = scores[scores.profile_id.isin(test_ids)]
    model_targets = {
        "M1-primary": "m1_primary",
        "M1-secondary": "m1_secondary",
        "M2-A": "m2a_primary",
        "M2-B": "m2b_primary",
        "Fusion": "all_fraud",
    }
    score_columns = {
        "M1-primary": "functional_risk_score",
        "M1-secondary": "functional_risk_score",
        "M2-A": "text_risk",
        "M2-B": "behavior_risk",
        "Fusion": "fusion_risk",
    }
    rows = []
    attribution = []
    for model, target_col in model_targets.items():
        score_col = score_columns[model]
        threshold = best_threshold(val_scores[target_col].astype(int).to_numpy(), val_scores[score_col].to_numpy())
        rows.append(metric_row(model, "scope_aligned", test_scores[target_col].astype(int).to_numpy(), test_scores[score_col].to_numpy(), threshold, len(test_scores)))
        for fraud_type, group in test_scores[test_scores.is_fraud.map(truthy)].groupby("fraud_type"):
            attribution.append({"model": model, "fraud_type": fraud_type, "count": len(group), "recall": float((group[score_col] >= threshold).mean()), "mean_score": float(group[score_col].mean())})
    metrics = pd.DataFrame(rows)
    attr = pd.DataFrame(attribution)
    metrics.to_csv(os.path.join(args.output_dir, "all_model_metrics.csv"), index=False)
    attr.to_csv(os.path.join(args.output_dir, "all_model_fraud_attribution.csv"), index=False)
    report = ["UNIFIED TEMPORAL MODEL EVALUATION", "=" * 78, f"Train: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}", "", "Scope-aligned metrics (threshold selected on validation, evaluated on test)", metrics.to_string(index=False), "", "All-fraud attribution on test", attr.to_string(index=False)]
    with open(os.path.join(args.output_dir, "all_model_evaluation_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()

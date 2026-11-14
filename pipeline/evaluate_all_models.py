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

from model1.evaluation.evaluate_m1 import load_profiles, run_scoring


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
    row["tn"] = int(tn)
    row["fp"] = int(fp)
    row["fn"] = int(fn)
    row["tp"] = int(tp)
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
    parser.add_argument("--m4_scores", default="./model4/outputs/m4_scores.csv")
    parser.add_argument("--m5_scores", default="./model5/outputs/m5_scores.csv")
    parser.add_argument("--m5_power", type=float, default=1.0,
                         help="Raise M5's graph_risk to this power before fusing (e.g. 2 shrinks mid-range scores toward 0, damping its pull on the noisy-OR fusion) without retraining anything.")
    parser.add_argument("--output_dir", default="./pipeline/outputs")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    has_m4 = bool(args.m4_scores) and os.path.exists(args.m4_scores)
    has_m5 = bool(args.m5_scores) and os.path.exists(args.m5_scores)
    print(f"M4 scores  : {args.m4_scores if has_m4 else 'not found, skipping M4 in fusion'}", flush=True)
    print(f"M5 scores  : {args.m5_scores if has_m5 else 'not found, skipping M5 in fusion'}" + (f" (power={args.m5_power})" if has_m5 else ""), flush=True)

    profile_df = pd.read_csv(args.profiles)
    split_dir = os.path.join(os.path.dirname(args.profiles), "splits")
    train = pd.read_csv(os.path.join(split_dir, "train.csv"))
    val = pd.read_csv(os.path.join(split_dir, "val.csv"))
    test = pd.read_csv(os.path.join(split_dir, "test.csv"))
    val_ids, test_ids = set(val["profile_id"]), set(test["profile_id"])
    target_map = targets(profile_df.set_index("profile_id").loc[profile_df["profile_id"]].reset_index())
    data = profile_df[["profile_id", "fraud_type", "is_fraud"]].copy()
    for name, target in target_map.items():
        data[name] = target.to_numpy()

    m1 = run_scoring(load_profiles(args.profiles))[["profile_id", "functional_risk_score"]]
    m2a = pd.read_csv(args.m2a_scores)[["profile_id", "text_risk"]]
    m2b = pd.read_csv(args.m2b_scores)[["profile_id", "behavior_risk"]]
    scores = data.merge(m1, on="profile_id").merge(m2a, on="profile_id").merge(m2b, on="profile_id")
    if has_m4:
        m4 = pd.read_csv(args.m4_scores)[["profile_id", "community_trust_risk"]]
        scores = scores.merge(m4, on="profile_id", how="left")
    else:
        scores["community_trust_risk"] = 0.0
    scores["community_trust_risk"] = scores["community_trust_risk"].fillna(0.0)

    if has_m5:
        m5 = pd.read_csv(args.m5_scores)[["profile_id", "graph_risk"]]
        scores = scores.merge(m5, on="profile_id", how="left")
    else:
        scores["graph_risk"] = 0.0
    scores["graph_risk"] = scores["graph_risk"].fillna(0.0)

    # Legit graph_risk on the test split, BEFORE fusing — same calibration
    # sanity check every model gets before entering a noisy-OR fusion: it
    # must sit near 0 for normal profiles, or it distorts the fusion the
    # way M2-A's uncalibrated baseline did.
    legit_test_graph_risk = scores[scores.profile_id.isin(test_ids) & ~scores["is_fraud"].map(truthy)]["graph_risk"]
    print(
        f"Legit graph_risk (test split, n={len(legit_test_graph_risk):,}): "
        f"mean={legit_test_graph_risk.mean():.4f}  p95={legit_test_graph_risk.quantile(0.95):.4f}",
        flush=True,
    )

    scores["graph_risk_fused"] = scores["graph_risk"] ** args.m5_power if has_m5 else 0.0
    base_terms = (1.0 - scores["functional_risk_score"]) * (1.0 - scores["text_risk"]) * (1.0 - scores["behavior_risk"]) * (1.0 - scores["community_trust_risk"])
    scores["fusion_risk_no_m5"] = 1.0 - base_terms
    scores["fusion_risk_with_m5"] = 1.0 - base_terms * (1.0 - scores["graph_risk_fused"])
    scores.to_csv(os.path.join(args.output_dir, "all_model_scores.csv"), index=False)

    val_scores = scores[scores.profile_id.isin(val_ids)]
    test_scores = scores[scores.profile_id.isin(test_ids)]
    model_targets = {
        "M1-primary": "m1_primary",
        "M1-secondary": "m1_secondary",
        "M2-A": "m2a_primary",
        "M2-B": "m2b_primary",
    }
    score_columns = {
        "M1-primary": "functional_risk_score",
        "M1-secondary": "functional_risk_score",
        "M2-A": "text_risk",
        "M2-B": "behavior_risk",
    }
    if has_m4:
        # M4 targets fraud generically (community reports aren't tied to one
        # sub-type), unlike M1/M2's primary targets, so it's evaluated
        # against the same all_fraud target as Fusion.
        model_targets["M4"] = "all_fraud"
        score_columns["M4"] = "community_trust_risk"
    if has_m5:
        # Same reasoning as M4 — M5 targets fraud generically, not one
        # sub-type, and is evaluated with its raw (unpowered) graph_risk;
        # m5_power only affects the fusion columns below.
        model_targets["M5"] = "all_fraud"
        score_columns["M5"] = "graph_risk"
    model_targets["Fusion-noM5"] = "all_fraud"
    score_columns["Fusion-noM5"] = "fusion_risk_no_m5"
    if has_m5:
        model_targets["Fusion-M5"] = "all_fraud"
        score_columns["Fusion-M5"] = "fusion_risk_with_m5"
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

    report = ["UNIFIED TEMPORAL MODEL EVALUATION", "=" * 78, f"Train: {len(train):,} | Validation: {len(val):,} | Test: {len(test):,}", "", "Scope-aligned metrics (threshold selected on validation, evaluated on test)", metrics.to_string(index=False), "", "All-fraud attribution on test (includes coordinated_ring)", attr.to_string(index=False)]

    if has_m5:
        no_m5_row = metrics[metrics.model == "Fusion-noM5"].iloc[0]
        with_m5_row = metrics[metrics.model == "Fusion-M5"].iloc[0]
        fp_delta = int(with_m5_row["fp"]) - int(no_m5_row["fp"])
        fp_relative = fp_delta / max(int(no_m5_row["fp"]), 1)
        report += [
            "",
            "False positives at each fusion's own validation-selected threshold",
            f"  Fusion-noM5: threshold={no_m5_row['threshold']:.2f}  FP={int(no_m5_row['fp'])}  (TP={int(no_m5_row['tp'])}, FN={int(no_m5_row['fn'])})",
            f"  Fusion-M5  : threshold={with_m5_row['threshold']:.2f}  FP={int(with_m5_row['fp'])}  (TP={int(with_m5_row['tp'])}, FN={int(with_m5_row['fn'])})",
            f"  Delta: {fp_delta:+d} FP ({fp_relative:+.1%} relative to Fusion-noM5)",
        ]

    report_text = "\n".join(report)
    with open(os.path.join(args.output_dir, "all_model_evaluation_report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report_text)
    print(report_text)


if __name__ == "__main__":
    main()

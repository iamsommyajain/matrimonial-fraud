"""
evaluate_m1.py

Runs Model 1 against your full dataset and prints:
  - Precision, Recall, F1 at multiple thresholds
  - AUC-ROC
  - Per fraud-type breakdown (which types M1 catches best)
  - Confusion matrix at threshold 0.55

Run from the model1/ directory:
    python evaluate_m1.py --dataset ../output/profiles.csv

Or with a custom threshold:
    python evaluate_m1.py --dataset ../output/profiles.csv --threshold 0.50
"""

import argparse
import sys
import os
import json

import pandas as pd
import numpy as np

# Allow running from model1/ directory
sys.path.insert(0, os.path.dirname(__file__))
from model1 import score_profile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_profiles(csv_path: str) -> list:
    """Load profiles.csv and return as list of dicts."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # JSON-encoded list columns — decode them back
    list_cols = ["face_embeddings", "exif_data", "login_timestamps",
                 "edit_timestamps", "photo_upload_dates", "login_ip_list",
                 "hobbies", "injected_signals"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else []
            )

    print(f"Loaded {len(df):,} profiles.")
    return df.to_dict(orient="records")


def run_scoring(profiles: list) -> pd.DataFrame:
    """Score all profiles and return a results DataFrame."""
    print(f"Scoring {len(profiles):,} profiles through M1...")
    results = []
    for i, p in enumerate(profiles):
        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {len(profiles):,} done...")
        r = score_profile(p)
        results.append({
            "profile_id":            r["profile_id"],
            "functional_risk_score": r["functional_risk_score"],
            "risk_level":            r["risk_level"],
            "flags":                 "|".join(r["flags"]),
            "is_fraud":              p.get("is_fraud", False),
            "fraud_type":            p.get("fraud_type") or "legitimate",
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall_f1(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1, tp, fp, fn, tn


def auc_roc(y_true, y_scores):
    """Simple trapezoid AUC — avoids sklearn dependency."""
    paired = sorted(zip(y_scores, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tps, fps = 0, 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0

    for score, label in paired:
        if label == 1:
            tps += 1
        else:
            fps += 1
            auc += (tps + prev_tp) / 2.0
            prev_tp = tps
        prev_fp = fps

    auc /= (n_pos * n_neg)
    return round(auc, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results_df: pd.DataFrame, threshold: float):
    y_true   = results_df["is_fraud"].astype(int).tolist()
    y_scores = results_df["functional_risk_score"].tolist()
    y_pred   = [1 if s >= threshold else 0 for s in y_scores]

    print("\n" + "=" * 60)
    print("MODEL 1 — FUNCTIONAL CONSISTENCY  |  EVALUATION REPORT")
    print("=" * 60)
    print(f"Dataset size : {len(results_df):,} profiles")
    print(f"Fraud rate   : {sum(y_true) / len(y_true) * 100:.1f}%")
    print(f"Threshold    : {threshold}")

    # ── Overall metrics ───────────────────────────────────────────────────
    prec, rec, f1, tp, fp, fn, tn = precision_recall_f1(y_true, y_pred)
    auc = auc_roc(y_true, y_scores)

    print(f"\n{'─'*40}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Precision : {prec:.4f}   ({tp} true positives, {fp} false positives)")
    print(f"  Recall    : {rec:.4f}   ({fn} missed fraud profiles)")
    print(f"  F1 Score  : {f1:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    print(f"\nConfusion Matrix (threshold={threshold}):")
    print(f"                 Predicted Clean   Predicted Fraud")
    print(f"  Actual Clean        {tn:>6}              {fp:>6}")
    print(f"  Actual Fraud        {fn:>6}              {tp:>6}")

    # ── Threshold sweep ───────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("Threshold Sweep:")
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    for t in [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80]:
        yp = [1 if s >= t else 0 for s in y_scores]
        p, r, f, *_ = precision_recall_f1(y_true, yp)
        marker = "  ← current" if t == threshold else ""
        print(f"  {t:>10.2f}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}{marker}")

    # ── Per fraud-type breakdown ───────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("Recall by Fraud Type (how many of each type M1 catches):")
    fraud_types = results_df[results_df["is_fraud"]]["fraud_type"].unique()
    for ft in sorted(fraud_types):
        subset = results_df[results_df["fraud_type"] == ft]
        caught = (subset["functional_risk_score"] >= threshold).sum()
        total  = len(subset)
        pct    = caught / total * 100 if total > 0 else 0
        print(f"  {ft:<25}  {caught:>4} / {total:<4}  ({pct:.0f}%)")

    # ── Score distribution ────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("Score Distribution:")
    fraud_scores  = results_df[results_df["is_fraud"]]["functional_risk_score"]
    legit_scores  = results_df[~results_df["is_fraud"]]["functional_risk_score"]
    print(f"  Fraud profiles  — mean: {fraud_scores.mean():.3f}  "
          f"median: {fraud_scores.median():.3f}  "
          f"p90: {fraud_scores.quantile(0.90):.3f}")
    print(f"  Legit profiles  — mean: {legit_scores.mean():.3f}  "
          f"median: {legit_scores.median():.3f}  "
          f"p90: {legit_scores.quantile(0.90):.3f}")

    # ── Most common flags ─────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("Most Common Flags (fraud profiles only):")
    fraud_flags = results_df[results_df["is_fraud"]]["flags"]
    from collections import Counter
    flag_counts = Counter()
    for flags_str in fraud_flags:
        if flags_str:
            for f in flags_str.split("|"):
                if f:
                    flag_counts[f] += 1
    for flag, count in flag_counts.most_common(10):
        print(f"  {flag:<35}  {count:>5}")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model 1 — Functional Consistency")
    parser.add_argument(
        "--dataset",   type=str, default="../output/profiles.csv",
        help="Path to profiles.csv"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Risk score threshold for fraud classification (default: 0.55)"
    )
    parser.add_argument(
        "--save",      type=str, default=None,
        help="Optional path to save results CSV (e.g. ../output/m1_scores.csv)"
    )
    args = parser.parse_args()

    profiles   = load_profiles(args.dataset)
    results_df = run_scoring(profiles)

    print_report(results_df, args.threshold)

    if args.save:
        results_df.to_csv(args.save, index=False)
        print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()

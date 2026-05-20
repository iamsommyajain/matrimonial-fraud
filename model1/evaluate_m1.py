"""
Evaluate Model 1 as both a ranking system and a classifier.

No sklearn dependency.  Diagnostics include PR/ROC metrics, calibration,
threshold recommendations, rule analytics, error cohorts, and throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model1 import score_profile


JSON_LIST_COLUMNS = (
    "login_timestamps",
    "edit_timestamps",
    "photo_upload_dates",
    "login_ip_list",
    "hobbies",
    "injected_signals",
)


def load_profiles(csv_path: str, limit: int | None = None) -> list[dict]:
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    for col in JSON_LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_parse_json_list)

    print(f"Loaded {len(df):,} profiles.")
    return df.to_dict(orient="records")


def run_scoring(profiles: list[dict]) -> pd.DataFrame:
    print(f"Scoring {len(profiles):,} profiles through M1...")
    rows = []
    start = time.perf_counter()

    for i, profile in enumerate(profiles, start=1):
        if i % 5000 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i:,} / {len(profiles):,} done ({i / elapsed:,.0f} profiles/sec)...")

        t0 = time.perf_counter()
        result = score_profile(profile, include_profiling=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        breakdown = result["score_breakdown"]
        effective = breakdown["effective_scores"]
        profiling = result.get("profiling", {})

        row = {
            "profile_id": result["profile_id"],
            "functional_risk_score": result["functional_risk_score"],
            "risk_level": result["risk_level"],
            "flags": "|".join(result["flags"]),
            "is_fraud": _truthy(profile.get("is_fraud")),
            "fraud_type": profile.get("fraud_type") or "legitimate",
            "latency_ms": latency_ms,
            "extraction_ms": profiling.get("extraction_ms", 0.0),
            "scoring_ms": profiling.get("scoring_ms", 0.0),
            "average_rule_confidence": result["uncertainty_summary"]["average_rule_confidence"],
            "missingness_impact": result["uncertainty_summary"]["missingness_impact"],
        }

        for rule in result["rule_results"]:
            name = rule["rule"]
            row[f"rule_score__{name}"] = rule["score"]
            row[f"rule_confidence__{name}"] = rule["confidence"]
            row[f"rule_effective__{name}"] = effective.get(name, 0.0)
            row[f"rule_fired__{name}"] = int(rule["score"] >= 0.55 or effective.get(name, 0.0) >= 0.06)
            row[f"rule_runtime_ms__{name}"] = profiling.get("rule_runtimes_ms", {}).get(name, 0.0)

        rows.append(row)

    total = time.perf_counter() - start
    df = pd.DataFrame(rows)
    df.attrs["total_runtime_sec"] = total
    df.attrs["throughput_per_sec"] = len(rows) / total if total else 0.0
    return df


def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tp, fp, fn, tn


def auc_roc(y_true, y_scores) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = pd.Series(y_scores).rank(method="average").to_numpy()
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return round(float(auc), 4)


def pr_curve(y_true, y_scores, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for threshold in thresholds:
        pred = np.asarray(y_scores) >= threshold
        precision, recall, f1, tp, fp, fn, tn = precision_recall_f1(y_true, pred)
        rows.append({
            "threshold": round(float(threshold), 4),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })
    return pd.DataFrame(rows)


def auc_pr(pr_df: pd.DataFrame) -> float:
    ordered = pr_df.sort_values("recall")
    precision = ordered["precision"].to_numpy()
    recall = ordered["recall"].to_numpy()
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(precision, recall)
    else:
        area = sum(
            (precision[i] + precision[i - 1]) * (recall[i] - recall[i - 1]) / 2
            for i in range(1, len(precision))
        )
    return round(float(area), 4)


def threshold_recommendations(pr_df: pd.DataFrame, precision_target: float,
                              recall_target: float) -> dict[str, Any]:
    best_f1 = pr_df.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    precision_ok = pr_df[pr_df["precision"] >= precision_target]
    recall_ok = pr_df[pr_df["recall"] >= recall_target]
    return {
        "best_f1": best_f1.to_dict(),
        "precision_constrained": (
            precision_ok.sort_values(["recall", "f1"], ascending=False).iloc[0].to_dict()
            if not precision_ok.empty else None
        ),
        "recall_constrained": (
            recall_ok.sort_values(["precision", "f1"], ascending=False).iloc[0].to_dict()
            if not recall_ok.empty else None
        ),
    }


def calibration_buckets(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    bucket = pd.cut(
        df["functional_risk_score"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    return (
        df.assign(bucket=bucket)
        .groupby("bucket", observed=False)
        .agg(
            count=("profile_id", "count"),
            avg_score=("functional_risk_score", "mean"),
            fraud_rate=("is_fraud", "mean"),
            avg_confidence=("average_rule_confidence", "mean"),
        )
        .reset_index()
    )


def ks_statistic(df: pd.DataFrame) -> float:
    fraud = np.sort(df.loc[df["is_fraud"], "functional_risk_score"].to_numpy())
    legit = np.sort(df.loc[~df["is_fraud"], "functional_risk_score"].to_numpy())
    if len(fraud) == 0 or len(legit) == 0:
        return 0.0
    values = np.sort(df["functional_risk_score"].unique())
    fraud_cdf = np.searchsorted(fraud, values, side="right") / len(fraud)
    legit_cdf = np.searchsorted(legit, values, side="right") / len(legit)
    return round(float(np.max(np.abs(fraud_cdf - legit_cdf))), 4)


def decile_lift(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values("functional_risk_score", ascending=False).copy()
    ranked["decile"] = pd.qcut(np.arange(len(ranked)), 10, labels=False, duplicates="drop") + 1
    base_rate = ranked["is_fraud"].mean()
    out = ranked.groupby("decile").agg(
        count=("profile_id", "count"),
        avg_score=("functional_risk_score", "mean"),
        fraud_rate=("is_fraud", "mean"),
        frauds=("is_fraud", "sum"),
    ).reset_index()
    out["lift"] = out["fraud_rate"] / base_rate if base_rate else 0.0
    return out


def percentile_overlap(df: pd.DataFrame) -> dict[str, float]:
    fraud = df.loc[df["is_fraud"], "functional_risk_score"]
    legit = df.loc[~df["is_fraud"], "functional_risk_score"]
    if fraud.empty or legit.empty:
        return {}
    legit_p90 = float(legit.quantile(0.90))
    fraud_below_legit_p90 = float((fraud <= legit_p90).mean())
    return {
        "fraud_p10": round(float(fraud.quantile(0.10)), 4),
        "fraud_p50": round(float(fraud.quantile(0.50)), 4),
        "fraud_p90": round(float(fraud.quantile(0.90)), 4),
        "legit_p10": round(float(legit.quantile(0.10)), 4),
        "legit_p50": round(float(legit.quantile(0.50)), 4),
        "legit_p90": round(legit_p90, 4),
        "fraud_share_at_or_below_legit_p90": round(fraud_below_legit_p90, 4),
    }


def rule_analytics(df: pd.DataFrame, threshold: float) -> dict[str, pd.DataFrame]:
    rule_cols = sorted(c for c in df.columns if c.startswith("rule_fired__"))
    rows = []
    for fired_col in rule_cols:
        rule = fired_col.replace("rule_fired__", "")
        score_col = f"rule_score__{rule}"
        eff_col = f"rule_effective__{rule}"
        fired = df[fired_col].astype(bool)
        fraud_fired = df.loc[df["is_fraud"], fired_col].mean() if df["is_fraud"].any() else 0.0
        legit_fired = df.loc[~df["is_fraud"], fired_col].mean() if (~df["is_fraud"]).any() else 0.0
        fp = df[(~df["is_fraud"]) & (df["functional_risk_score"] >= threshold)]
        fn = df[(df["is_fraud"]) & (df["functional_risk_score"] < threshold)]
        rows.append({
            "rule": rule,
            "firing_rate": fired.mean(),
            "fraud_firing_rate": fraud_fired,
            "legit_firing_rate": legit_fired,
            "separation": fraud_fired - legit_fired,
            "avg_score": df[score_col].mean(),
            "avg_effective_contribution": df[eff_col].mean(),
            "fp_firing_rate": fp[fired_col].mean() if not fp.empty else 0.0,
            "fn_firing_rate": fn[fired_col].mean() if not fn.empty else 0.0,
        })

    analytics = pd.DataFrame(rows).sort_values("avg_effective_contribution", ascending=False)
    low_info = analytics[
        (analytics["firing_rate"] < 0.01)
        | (analytics["avg_effective_contribution"] < 0.001)
        | (analytics["separation"].abs() < 0.01)
    ].sort_values("avg_effective_contribution")

    fp_heavy = analytics[
        (analytics["fp_firing_rate"] > analytics["fraud_firing_rate"])
        & (analytics["fp_firing_rate"] > 0)
    ].sort_values("fp_firing_rate", ascending=False)

    correlated = correlated_rules(df)
    return {"summary": analytics, "low_info": low_info, "fp_heavy": fp_heavy, "correlated": correlated}


def correlated_rules(df: pd.DataFrame, min_abs_corr: float = 0.75) -> pd.DataFrame:
    eff_cols = sorted(c for c in df.columns if c.startswith("rule_effective__"))
    rows = []
    for left, right in combinations(eff_cols, 2):
        if df[left].std() == 0 or df[right].std() == 0:
            continue
        corr = df[left].corr(df[right])
        if abs(corr) >= min_abs_corr:
            rows.append({
                "rule_a": left.replace("rule_effective__", ""),
                "rule_b": right.replace("rule_effective__", ""),
                "corr": corr,
            })
    return pd.DataFrame(rows).sort_values("corr", key=lambda s: s.abs(), ascending=False) if rows else pd.DataFrame()


def rule_runtime_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in sorted(c for c in df.columns if c.startswith("rule_runtime_ms__")):
        rows.append({
            "rule": col.replace("rule_runtime_ms__", ""),
            "avg_runtime_ms": df[col].mean(),
            "p95_runtime_ms": df[col].quantile(0.95),
        })
    return pd.DataFrame(rows).sort_values("avg_runtime_ms", ascending=False)


def error_cohorts(df: pd.DataFrame, threshold: float) -> dict[str, pd.DataFrame]:
    fp = df[(~df["is_fraud"]) & (df["functional_risk_score"] >= threshold)]
    fn = df[(df["is_fraud"]) & (df["functional_risk_score"] < threshold)]
    return {
        "false_positive_flags": _flag_counts(fp),
        "false_negative_fraud_types": (
            fn.groupby("fraud_type")
            .agg(count=("profile_id", "count"), avg_score=("functional_risk_score", "mean"),
                 avg_missingness=("missingness_impact", "mean"))
            .sort_values("count", ascending=False)
            .reset_index()
        ),
        "highest_false_positives": fp.sort_values("functional_risk_score", ascending=False).head(10),
        "lowest_false_negatives": fn.sort_values("functional_risk_score").head(10),
    }


def print_report(results_df: pd.DataFrame, threshold: float,
                 precision_target: float, recall_target: float) -> None:
    y_true = results_df["is_fraud"].astype(int).to_numpy()
    y_scores = results_df["functional_risk_score"].to_numpy()
    y_pred = y_scores >= threshold
    precision, recall, f1, tp, fp, fn, tn = precision_recall_f1(y_true, y_pred)
    pr_df = pr_curve(y_true, y_scores)
    recs = threshold_recommendations(pr_df, precision_target, recall_target)
    rules = rule_analytics(results_df, threshold)
    cohorts = error_cohorts(results_df, threshold)

    print("\n" + "=" * 72)
    print("MODEL 1 - FUNCTIONAL CONSISTENCY | DIAGNOSTIC REPORT")
    print("=" * 72)
    print(f"Dataset size : {len(results_df):,} profiles")
    print(f"Fraud rate   : {y_true.mean() * 100:.1f}%")
    print(f"Threshold    : {threshold:.2f}")
    print(f"Throughput   : {results_df.attrs.get('throughput_per_sec', 0):,.0f} profiles/sec")
    print(f"Latency      : mean {results_df['latency_ms'].mean():.3f} ms | p95 {results_df['latency_ms'].quantile(0.95):.3f} ms")
    print(f"Extraction   : mean {results_df['extraction_ms'].mean():.3f} ms")
    print(f"Scoring      : mean {results_df['scoring_ms'].mean():.3f} ms")

    print("\nOverall Classification")
    print(f"  AUC-ROC   : {auc_roc(y_true, y_scores):.4f}")
    print(f"  AUC-PR    : {auc_pr(pr_df):.4f}")
    print(f"  KS stat   : {ks_statistic(results_df):.4f}")
    print(f"  Precision : {precision:.4f} ({tp} TP, {fp} FP)")
    print(f"  Recall    : {recall:.4f} ({fn} missed fraud)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nThreshold Recommendations")
    _print_threshold_row("Best F1", recs["best_f1"])
    _print_threshold_row(f"Precision >= {precision_target:.2f}", recs["precision_constrained"])
    _print_threshold_row(f"Recall >= {recall_target:.2f}", recs["recall_constrained"])

    print("\nThreshold Sweep")
    print(pr_df[pr_df["threshold"].isin([0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80])]
          [["threshold", "precision", "recall", "f1", "tp", "fp", "fn"]]
          .to_string(index=False, formatters=_formatters()))

    print("\nCalibration Buckets")
    print(calibration_buckets(results_df)[["bucket", "count", "avg_score", "fraud_rate", "avg_confidence"]]
          .to_string(index=False, formatters=_formatters()))

    print("\nDecile Lift (1 = highest-risk decile)")
    print(decile_lift(results_df).head(10).to_string(index=False, formatters=_formatters()))

    print("\nPercentile Overlap")
    for key, value in percentile_overlap(results_df).items():
        print(f"  {key:<36} {value:.4f}")

    print("\nRecall by Fraud Type")
    fraud_rows = []
    for ft, subset in results_df[results_df["is_fraud"]].groupby("fraud_type"):
        caught = int((subset["functional_risk_score"] >= threshold).sum())
        total = len(subset)
        fraud_rows.append({"fraud_type": ft, "caught": caught, "total": total,
                           "recall": caught / total if total else 0.0,
                           "avg_score": subset["functional_risk_score"].mean()})
    print(pd.DataFrame(fraud_rows).sort_values("recall").to_string(index=False, formatters=_formatters()))

    print("\nRule Contribution Analytics")
    print(rules["summary"].head(15).to_string(index=False, formatters=_formatters()))

    print("\nRule Runtime Hotspots")
    print(rule_runtime_hotspots(results_df).head(12).to_string(index=False, formatters=_formatters()))

    print("\nFalse-Positive-Heavy Rules")
    print(_empty_or_table(rules["fp_heavy"].head(10), formatters=_formatters()))

    print("\nLow-Information Rules")
    print(_empty_or_table(rules["low_info"].head(10), formatters=_formatters()))

    print("\nRedundant/Correlated Rule Pairs")
    print(_empty_or_table(rules["correlated"].head(10), formatters=_formatters()))

    print("\nFalse Positive Flag Cohorts")
    print(_empty_or_table(cohorts["false_positive_flags"].head(10), formatters=_formatters()))

    print("\nFalse Negative Cohorts by Fraud Type")
    print(_empty_or_table(cohorts["false_negative_fraud_types"].head(10), formatters=_formatters()))
    print("\n" + "=" * 72)


def _parse_json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not pd.isna(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _flag_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = Counter()
    for flags in df.get("flags", pd.Series(dtype=str)).fillna(""):
        for flag in str(flags).split("|"):
            if flag:
                counts[flag] += 1
    return pd.DataFrame([{"flag": k, "count": v} for k, v in counts.most_common()])


def _print_threshold_row(label: str, row: dict | None) -> None:
    if row is None:
        print(f"  {label:<20} no threshold satisfies constraint")
        return
    print(
        f"  {label:<20} threshold={row['threshold']:.2f} "
        f"precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}"
    )


def _empty_or_table(df: pd.DataFrame, formatters=None) -> str:
    if df is None or df.empty:
        return "  none"
    return df.to_string(index=False, formatters=formatters)


def _formatters() -> dict:
    return {
        col: "{:.4f}".format for col in [
            "precision", "recall", "f1", "avg_score", "fraud_rate",
            "avg_confidence", "lift", "firing_rate", "fraud_firing_rate",
            "legit_firing_rate", "separation", "avg_effective_contribution",
            "fp_firing_rate", "fn_firing_rate", "avg_missingness", "corr",
            "avg_runtime_ms", "p95_runtime_ms",
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model 1 functional consistency")
    parser.add_argument("--dataset", type=str, default="../output/profiles.csv")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--precision-target", type=float, default=0.80)
    parser.add_argument("--recall-target", type=float, default=0.80)
    args = parser.parse_args()

    profiles = load_profiles(args.dataset, limit=args.limit)
    results_df = run_scoring(profiles)
    print_report(results_df, args.threshold, args.precision_target, args.recall_target)

    if args.save:
        results_df.to_csv(args.save, index=False)
        print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()

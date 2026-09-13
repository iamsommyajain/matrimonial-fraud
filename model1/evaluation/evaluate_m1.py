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

# Support both package execution and direct execution from dataset_generation/.
MODEL1_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_GENERATION_DIR = os.path.dirname(MODEL1_DIR)
sys.path.insert(0, DATASET_GENERATION_DIR)
try:
    from ..api.model1 import score_profile
except ImportError:
    from model1.api.model1 import score_profile


JSON_LIST_COLUMNS = (
    "login_timestamps",
    "edit_timestamps",
    "photo_upload_dates",
    "login_ip_list",
    "hobbies",
    "injected_signals",
)

FUNCTIONAL_FRAUD_TYPES = frozenset({"functional", "multi"})
PARTIAL_FRAUD_TYPES = frozenset({"multi"})
M1_SCOPE_EXPECTATION = {
    "functional": ("YES", "High"),
    "multi": ("PARTIAL", "Medium"),
    "template_bio": ("NO", "Low"),
    "coordinated_ring": ("NO", "Low"),
    "financial_scam": ("NO", "Low"),
    "legitimate": ("N/A", "N/A"),
}


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
        fusion = breakdown.get("fusion_diagnostics", {})
        profiling = result.get("profiling", {})

        fraud_type = _clean_fraud_type(profile.get("fraud_type"))
        row = {
            "profile_id": result["profile_id"],
            "functional_risk_score": result["functional_risk_score"],
            "risk_level": result["risk_level"],
            "flags": "|".join(result["flags"]),
            "is_fraud": _truthy(profile.get("is_fraud")),
            "fraud_type": fraud_type,
            "latency_ms": latency_ms,
            "extraction_ms": profiling.get("extraction_ms", 0.0),
            "scoring_ms": profiling.get("scoring_ms", 0.0),
            "average_rule_confidence": result["uncertainty_summary"]["average_rule_confidence"],
            "missingness_impact": result["uncertainty_summary"]["missingness_impact"],
            "uncertainty_only_score": result["uncertainty_summary"].get("uncertainty_only_score", 0.0),
            "suspicious_missingness_score": result["uncertainty_summary"].get("suspicious_missingness_score", 0.0),
            "pre_fusion_score": fusion.get("pre_fusion_score", 0.0),
            "post_fusion_score": fusion.get("post_fusion_score", result["functional_risk_score"]),
            "damping_ratio": fusion.get("damping_ratio", 0.0),
            "number_of_active_rules": fusion.get("number_of_active_rules", 0),
            "contribution_entropy": fusion.get("contribution_entropy", 0.0),
            "score_space_utilization": fusion.get("score_space_utilization", 0.0),
        }
        row["is_functional_fraud"] = row["fraud_type"] in FUNCTIONAL_FRAUD_TYPES
        row["is_m1_eval_row"] = row["fraud_type"] == "legitimate" or row["is_functional_fraud"]

        for rule in result["rule_results"]:
            name = rule["rule"]
            row[f"rule_score__{name}"] = rule["score"]
            row[f"rule_confidence__{name}"] = rule["confidence"]
            row[f"rule_effective__{name}"] = effective.get(name, 0.0)
            row[f"rule_fired__{name}"] = int(rule.get("fired") or rule["score"] >= 0.55 or effective.get(name, 0.0) >= 0.06)
            row[f"rule_runtime_ms__{name}"] = profiling.get("rule_runtimes_ms", {}).get(name, 0.0)
            row[f"rule_raw__{name}"] = _numeric_or_nan(rule.get("raw_signal_value"))
            row[f"rule_normalized__{name}"] = rule.get("normalized_signal_value")
            row[f"rule_before_fusion__{name}"] = rule.get("contribution_before_fusion", 0.0)
            row[f"rule_after_fusion__{name}"] = rule.get("contribution_after_fusion", 0.0)
            row[f"rule_abstained_reason__{name}"] = rule.get("abstained_reason") or ""
            row[f"rule_missing_fields__{name}"] = "|".join(rule.get("missing_fields") or [])
            row[f"rule_missingness_penalty__{name}"] = rule.get("missingness_penalty", 0.0)

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
                              recall_target: float, total_profiles: int | None = None,
                              fraud_queue_budget: int | None = None) -> dict[str, Any]:
    best_f1 = pr_df.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    precision_ok = pr_df[pr_df["precision"] >= precision_target]
    recall_ok = pr_df[pr_df["recall"] >= recall_target]
    budget_row = None
    if total_profiles and fraud_queue_budget:
        budget_ok = pr_df[pr_df["tp"] + pr_df["fp"] <= fraud_queue_budget]
        if not budget_ok.empty:
            budget_row = budget_ok.sort_values(["recall", "precision"], ascending=False).iloc[0].to_dict()
    return {
        "max_f1": best_f1.to_dict(),
        "best_f1": best_f1.to_dict(),
        "max_recall_at_precision": (
            precision_ok.sort_values(["recall", "f1"], ascending=False).iloc[0].to_dict()
            if not precision_ok.empty else None
        ),
        "max_precision_at_recall": (
            recall_ok.sort_values(["precision", "f1"], ascending=False).iloc[0].to_dict()
            if not recall_ok.empty else None
        ),
        "precision_constrained": (
            precision_ok.sort_values(["recall", "f1"], ascending=False).iloc[0].to_dict()
            if not precision_ok.empty else None
        ),
        "recall_constrained": (
            recall_ok.sort_values(["precision", "f1"], ascending=False).iloc[0].to_dict()
            if not recall_ok.empty else None
        ),
        "fraud_queue_budget": budget_row,
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


def target_calibration_buckets(df: pd.DataFrame, target_col: str, bins: int = 10) -> pd.DataFrame:
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
            target_rate=(target_col, "mean"),
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


def target_ks_statistic(df: pd.DataFrame, target_col: str) -> float:
    pos = np.sort(df.loc[df[target_col], "functional_risk_score"].to_numpy())
    neg = np.sort(df.loc[~df[target_col], "functional_risk_score"].to_numpy())
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    values = np.sort(df["functional_risk_score"].unique())
    pos_cdf = np.searchsorted(pos, values, side="right") / len(pos)
    neg_cdf = np.searchsorted(neg, values, side="right") / len(neg)
    return round(float(np.max(np.abs(pos_cdf - neg_cdf))), 4)


def cohen_d_scores(df: pd.DataFrame, target_col: str) -> float:
    pos = df.loc[df[target_col], "functional_risk_score"].to_numpy()
    neg = df.loc[~df[target_col], "functional_risk_score"].to_numpy()
    if len(pos) < 2 or len(neg) < 2:
        return 0.0
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2))
    return round(float((pos.mean() - neg.mean()) / pooled), 4) if pooled else 0.0


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


def target_decile_lift(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    ranked = df.sort_values("functional_risk_score", ascending=False).copy()
    ranked["decile"] = pd.qcut(np.arange(len(ranked)), 10, labels=False, duplicates="drop") + 1
    base_rate = ranked[target_col].mean()
    out = ranked.groupby("decile").agg(
        count=("profile_id", "count"),
        avg_score=("functional_risk_score", "mean"),
        target_rate=(target_col, "mean"),
        targets=(target_col, "sum"),
    ).reset_index()
    out["lift"] = out["target_rate"] / base_rate if base_rate else 0.0
    out["capture_rate"] = out["targets"] / ranked[target_col].sum() if ranked[target_col].sum() else 0.0
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


def target_percentile_overlap(df: pd.DataFrame, target_col: str) -> dict[str, float]:
    pos = df.loc[df[target_col], "functional_risk_score"]
    neg = df.loc[~df[target_col], "functional_risk_score"]
    if pos.empty or neg.empty:
        return {}
    neg_p90 = float(neg.quantile(0.90))
    return {
        "target_p10": round(float(pos.quantile(0.10)), 4),
        "target_p50": round(float(pos.quantile(0.50)), 4),
        "target_p90": round(float(pos.quantile(0.90)), 4),
        "negative_p10": round(float(neg.quantile(0.10)), 4),
        "negative_p50": round(float(neg.quantile(0.50)), 4),
        "negative_p90": round(neg_p90, 4),
        "target_share_at_or_below_negative_p90": round(float((pos <= neg_p90).mean()), 4),
    }


def score_histogram(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    fraud_counts, _ = np.histogram(df.loc[df["is_fraud"], "functional_risk_score"], bins=edges)
    legit_counts, _ = np.histogram(df.loc[~df["is_fraud"], "functional_risk_score"], bins=edges)
    rows = []
    for i in range(bins):
        rows.append({
            "score_bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
            "fraud_count": int(fraud_counts[i]),
            "legit_count": int(legit_counts[i]),
            "fraud_share": fraud_counts[i] / fraud_counts.sum() if fraud_counts.sum() else 0.0,
            "legit_share": legit_counts[i] / legit_counts.sum() if legit_counts.sum() else 0.0,
        })
    return pd.DataFrame(rows)


def target_score_histogram(df: pd.DataFrame, target_col: str, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    target_counts, _ = np.histogram(df.loc[df[target_col], "functional_risk_score"], bins=edges)
    neg_counts, _ = np.histogram(df.loc[~df[target_col], "functional_risk_score"], bins=edges)
    rows = []
    for i in range(bins):
        rows.append({
            "score_bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}",
            "target_count": int(target_counts[i]),
            "negative_count": int(neg_counts[i]),
            "target_share": target_counts[i] / target_counts.sum() if target_counts.sum() else 0.0,
            "negative_share": neg_counts[i] / neg_counts.sum() if neg_counts.sum() else 0.0,
        })
    return pd.DataFrame(rows)


def js_divergence_scores(df: pd.DataFrame, bins: int = 20) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    fraud, _ = np.histogram(df.loc[df["is_fraud"], "functional_risk_score"], bins=edges)
    legit, _ = np.histogram(df.loc[~df["is_fraud"], "functional_risk_score"], bins=edges)
    if fraud.sum() == 0 or legit.sum() == 0:
        return 0.0
    p = fraud / fraud.sum()
    q = legit / legit.sum()
    m = 0.5 * (p + q)
    return round(float(0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)), 4)


def target_js_divergence_scores(df: pd.DataFrame, target_col: str, bins: int = 20) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    pos, _ = np.histogram(df.loc[df[target_col], "functional_risk_score"], bins=edges)
    neg, _ = np.histogram(df.loc[~df[target_col], "functional_risk_score"], bins=edges)
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.0
    p = pos / pos.sum()
    q = neg / neg.sum()
    m = 0.5 * (p + q)
    return round(float(0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)), 4)


def top_k_precision(df: pd.DataFrame, ks=(100, 500, 1000)) -> pd.DataFrame:
    ranked = df.sort_values("functional_risk_score", ascending=False)
    rows = []
    for k in ks:
        subset = ranked.head(min(k, len(ranked)))
        rows.append({
            "top_k": len(subset),
            "precision": subset["is_fraud"].mean() if len(subset) else 0.0,
            "frauds_captured": int(subset["is_fraud"].sum()),
            "capture_rate": subset["is_fraud"].sum() / df["is_fraud"].sum() if df["is_fraud"].sum() else 0.0,
            "min_score": subset["functional_risk_score"].min() if len(subset) else 0.0,
        })
    return pd.DataFrame(rows)


def target_top_k(df: pd.DataFrame, target_col: str, ks=(100, 500, 1000)) -> pd.DataFrame:
    ranked = df.sort_values("functional_risk_score", ascending=False)
    total_targets = ranked[target_col].sum()
    rows = []
    for k in ks:
        subset = ranked.head(min(k, len(ranked)))
        rows.append({
            "top_k": len(subset),
            "precision_at_k": subset[target_col].mean() if len(subset) else 0.0,
            "targets_captured": int(subset[target_col].sum()),
            "recall_at_k": subset[target_col].sum() / total_targets if total_targets else 0.0,
            "min_score": subset["functional_risk_score"].min() if len(subset) else 0.0,
        })
    return pd.DataFrame(rows)


def tail_risk_concentration(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_fraud = df["is_fraud"].sum()
    for pct in (0.01, 0.02, 0.05, 0.10):
        n = max(1, int(len(df) * pct))
        tail = df.sort_values("functional_risk_score", ascending=False).head(n)
        rows.append({
            "tail_pct": pct,
            "profiles": n,
            "frauds": int(tail["is_fraud"].sum()),
            "capture_rate": tail["is_fraud"].sum() / total_fraud if total_fraud else 0.0,
            "precision": tail["is_fraud"].mean(),
        })
    return pd.DataFrame(rows)


def target_tail_risk_concentration(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    total_targets = df[target_col].sum()
    for pct in (0.01, 0.02, 0.05, 0.10):
        n = max(1, int(len(df) * pct))
        tail = df.sort_values("functional_risk_score", ascending=False).head(n)
        rows.append({
            "tail_pct": pct,
            "profiles": n,
            "targets": int(tail[target_col].sum()),
            "recall_at_tail": tail[target_col].sum() / total_targets if total_targets else 0.0,
            "precision": tail[target_col].mean(),
        })
    return pd.DataFrame(rows)


def rule_analytics(df: pd.DataFrame, threshold: float) -> dict[str, pd.DataFrame]:
    rule_cols = sorted(c for c in df.columns if c.startswith("rule_fired__"))
    rows = []
    for fired_col in rule_cols:
        rule = fired_col.replace("rule_fired__", "")
        score_col = f"rule_score__{rule}"
        eff_col = f"rule_effective__{rule}"
        before_col = f"rule_before_fusion__{rule}"
        after_col = f"rule_after_fusion__{rule}"
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
            "avg_before_fusion": df[before_col].mean() if before_col in df else 0.0,
            "avg_after_fusion": df[after_col].mean() if after_col in df else 0.0,
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
    abstentions = rule_abstention_report(df)
    distributions = rule_distribution_report(df)
    collapse = signal_collapse_report(df, analytics)
    interactions = interaction_analytics(df, threshold)
    return {
        "summary": analytics,
        "low_info": low_info,
        "fp_heavy": fp_heavy,
        "correlated": correlated,
        "abstentions": abstentions,
        "distributions": distributions,
        "collapse": collapse,
        "interactions": interactions,
    }


def rule_abstention_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in sorted(c for c in df.columns if c.startswith("rule_abstained_reason__")):
        rule = col.replace("rule_abstained_reason__", "")
        reasons = df[col].fillna("")
        abstained = reasons != ""
        top_reason = reasons[abstained].mode()
        missing_col = f"rule_missing_fields__{rule}"
        missing_counts = Counter()
        if missing_col in df:
            for value in df[missing_col].fillna(""):
                for field in str(value).split("|"):
                    if field:
                        missing_counts[field] += 1
        rows.append({
            "rule": rule,
            "abstain_rate": abstained.mean(),
            "top_abstain_reason": top_reason.iloc[0] if not top_reason.empty else "",
            "top_missing_features": ", ".join(f"{k}:{v}" for k, v in missing_counts.most_common(3)),
        })
    return pd.DataFrame(rows).sort_values("abstain_rate", ascending=False)


def rule_distribution_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in sorted(c for c in df.columns if c.startswith("rule_score__")):
        rule = col.replace("rule_score__", "")
        raw_col = f"rule_raw__{rule}"
        norm_col = f"rule_normalized__{rule}"
        conf_col = f"rule_confidence__{rule}"
        rows.append({
            "rule": rule,
            "raw_mean": df[raw_col].mean() if raw_col in df else np.nan,
            "raw_p95": df[raw_col].quantile(0.95) if raw_col in df else np.nan,
            "normalized_mean": df[norm_col].mean() if norm_col in df else np.nan,
            "normalized_p95": df[norm_col].quantile(0.95) if norm_col in df else np.nan,
            "confidence_mean": df[conf_col].mean() if conf_col in df else np.nan,
            "confidence_p10": df[conf_col].quantile(0.10) if conf_col in df else np.nan,
            "score_variance": df[col].var(),
        })
    return pd.DataFrame(rows).sort_values("normalized_mean", ascending=False)


def signal_collapse_report(df: pd.DataFrame, analytics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in analytics.iterrows():
        rule = row["rule"]
        score_col = f"rule_score__{rule}"
        eff_col = f"rule_effective__{rule}"
        flags = []
        if row["firing_rate"] < 0.005:
            flags.append("low_firing_rate")
        if df[score_col].var() < 1e-5:
            flags.append("near_zero_variance")
        if df[eff_col].mean() < 0.001:
            flags.append("tiny_contribution")
        if row["avg_score"] > 0.01 and abs(row["separation"]) < 0.01 and row.get("avg_effective_contribution", 0) > 0:
            flags.append("high_conf_low_separability")
        if flags:
            rows.append({
                "rule": rule,
                "flags": ",".join(flags),
                "firing_rate": row["firing_rate"],
                "separation": row["separation"],
                "avg_effective_contribution": row["avg_effective_contribution"],
                "score_variance": df[score_col].var(),
            })
    return pd.DataFrame(rows).sort_values("avg_effective_contribution", ascending=False) if rows else pd.DataFrame()


def interaction_analytics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    base_rate = df["is_fraud"].mean()
    for col in sorted(c for c in df.columns if c.startswith("rule_fired__interaction_")):
        rule = col.replace("rule_fired__", "")
        fired = df[col].astype(bool)
        fired_df = df[fired]
        fraud_rate = fired_df["is_fraud"].mean() if len(fired_df) else 0.0
        rows.append({
            "rule": rule,
            "firing_rate": fired.mean(),
            "fraud_rate_when_fired": fraud_rate,
            "interaction_lift": fraud_rate / base_rate if base_rate else 0.0,
            "avg_contribution": df[f"rule_effective__{rule}"].mean(),
            "fp_firing_rate": df[(~df["is_fraud"]) & (df["functional_risk_score"] >= threshold)][col].mean()
            if len(df[(~df["is_fraud"]) & (df["functional_risk_score"] >= threshold)]) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("interaction_lift", ascending=False) if rows else pd.DataFrame()


def rule_signal_validity(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    for col in sorted(c for c in df.columns if c.startswith("rule_fired__")):
        rule = col.replace("rule_fired__", "")
        fired = df[col].astype(bool)
        target = df[target_col].astype(bool)
        a = int((fired & target).sum())
        b = int((fired & ~target).sum())
        c = int((~fired & target).sum())
        d = int((~fired & ~target).sum())
        odds_ratio = (
            ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
            if (a + b) > 0 else 1.0
        )
        target_fire = fired[target].mean() if target.any() else 0.0
        neg_fire = fired[~target].mean() if (~target).any() else 0.0
        rows.append({
            "rule": rule,
            "target_firing_rate": target_fire,
            "negative_firing_rate": neg_fire,
            "separation": target_fire - neg_fire,
            "odds_ratio": odds_ratio,
            "mutual_info": mutual_information_binary(fired, target),
        })
    return pd.DataFrame(rows).sort_values("mutual_info", ascending=False)


def interpretability_quality(df: pd.DataFrame) -> dict[str, float]:
    after_cols = [c for c in df.columns if c.startswith("rule_after_fusion__")]
    if not after_cols:
        return {}
    contrib = df[after_cols].to_numpy(dtype=float)
    active_counts = (contrib > 0).sum(axis=1)
    totals = contrib.sum(axis=1)
    max_contrib = contrib.max(axis=1)
    dominant_share = np.divide(max_contrib, totals, out=np.zeros_like(max_contrib), where=totals > 0)
    conflicting_rate = ((df["number_of_active_rules"] >= 3) & (df["contribution_entropy"] > 0.75)).mean()
    return {
        "avg_contributing_rules": round(float(active_counts.mean()), 4),
        "p95_contributing_rules": round(float(np.quantile(active_counts, 0.95)), 4),
        "avg_dominant_rule_share": round(float(dominant_share.mean()), 4),
        "avg_explanation_entropy": round(float(df["contribution_entropy"].mean()), 4),
        "conflicting_evidence_rate": round(float(conflicting_rate), 4),
    }


def fraud_type_attribution_matrix(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for fraud_type, group in df.groupby("fraud_type"):
        intended, expected = M1_SCOPE_EXPECTATION.get(fraud_type, ("UNKNOWN", "UNKNOWN"))
        rows.append({
            "fraud_type": fraud_type,
            "intended_for_m1": intended,
            "expected_detection": expected,
            "count": len(group),
            "avg_score": group["functional_risk_score"].mean(),
            "p90_score": group["functional_risk_score"].quantile(0.90),
            "threshold_recall": (group["functional_risk_score"] >= threshold).mean(),
        })
    order = {"YES": 0, "PARTIAL": 1, "NO": 2, "N/A": 3, "UNKNOWN": 4}
    out = pd.DataFrame(rows)
    out["_order"] = out["intended_for_m1"].map(order).fillna(4)
    return out.sort_values(["_order", "fraud_type"]).drop(columns=["_order"])


def mutual_information_binary(x, y) -> float:
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    n = len(x)
    if n == 0:
        return 0.0
    mi = 0.0
    for xv in (False, True):
        for yv in (False, True):
            pxy = ((x == xv) & (y == yv)).sum() / n
            if pxy == 0:
                continue
            px = (x == xv).sum() / n
            py = (y == yv).sum() / n
            mi += pxy * np.log2(pxy / (px * py))
    return round(float(mi), 6)


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


def fusion_saturation_report(df: pd.DataFrame) -> dict[str, float]:
    before = df["pre_fusion_score"].replace(0, np.nan)
    damping = df["post_fusion_score"] / before
    clipping = (df["score_space_utilization"] >= 0.98).mean()
    return {
        "avg_pre_fusion_score": round(float(df["pre_fusion_score"].mean()), 4),
        "avg_post_fusion_score": round(float(df["post_fusion_score"].mean()), 4),
        "avg_damping_factor": round(float(damping.fillna(0).mean()), 4),
        "avg_active_rules": round(float(df["number_of_active_rules"].mean()), 4),
        "avg_contribution_entropy": round(float(df["contribution_entropy"].mean()), 4),
        "score_space_utilization_p95": round(float(df["score_space_utilization"].quantile(0.95)), 4),
        "contribution_clipping_rate": round(float(clipping), 4),
    }


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


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


def print_concise_report(results_df: pd.DataFrame, threshold: float,
                         precision_target: float, recall_target: float,
                         fraud_queue_budget: int | None = None) -> None:
    m1_df = results_df[results_df["is_m1_eval_row"]].copy()
    functional_y = m1_df["is_functional_fraud"].astype(int).to_numpy()
    functional_scores = m1_df["functional_risk_score"].to_numpy()
    functional_pred = functional_scores >= threshold
    f_prec, f_rec, f_f1, f_tp, f_fp, f_fn, f_tn = precision_recall_f1(
        functional_y, functional_pred
    )
    functional_pr = pr_curve(functional_y, functional_scores)
    recommendations = threshold_recommendations(
        functional_pr,
        precision_target,
        recall_target,
        len(m1_df),
        fraud_queue_budget,
    )

    print("\n" + "=" * 60)
    print("MODEL 1 - FUNCTIONAL CONSISTENCY REPORT")
    print("=" * 60)
    print(f"Dataset    : {len(results_df):,} profiles")
    print(f"Threshold  : {threshold:.2f}")
    print(f"Throughput : {results_df.attrs.get('throughput_per_sec', 0):,.0f} profiles/sec")
    print("\nPrimary metrics")
    print(f"  AUC-ROC   : {auc_roc(functional_y, functional_scores):.4f}")
    print(f"  AUC-PR    : {auc_pr(functional_pr):.4f}")
    print(f"  Precision : {f_prec:.4f} ({f_tp} TP, {f_fp} FP)")
    print(f"  Recall    : {f_rec:.4f} ({f_fn} missed)")
    print(f"  F1 Score  : {f_f1:.4f}")
    print(f"  Confusion : TN={f_tn}, FP={f_fp}, FN={f_fn}, TP={f_tp}")

    print("\nThreshold recommendations")
    _print_threshold_row("Best F1", recommendations["best_f1"])
    _print_threshold_row("Precision target", recommendations["precision_constrained"])
    _print_threshold_row("Recall target", recommendations["recall_constrained"])
    _print_threshold_row("Review budget", recommendations["fraud_queue_budget"])

    print("\nRecall by fraud type")
    fraud_rows = []
    for fraud_type, subset in results_df[results_df["is_fraud"]].groupby("fraud_type"):
        caught = int((subset["functional_risk_score"] >= threshold).sum())
        total = len(subset)
        fraud_rows.append({
            "fraud_type": fraud_type,
            "caught": caught,
            "total": total,
            "recall": caught / total if total else 0.0,
        })
    if fraud_rows:
        print(pd.DataFrame(fraud_rows).sort_values("recall").to_string(index=False))
    else:
        print("  no fraud rows")
    print("\nDetailed rule diagnostics are available in the saved CSV.")
    print("=" * 60)


def print_report(results_df: pd.DataFrame, threshold: float,
                 precision_target: float, recall_target: float,
                 fraud_queue_budget: int | None = None) -> None:
    m1_df = results_df[results_df["is_m1_eval_row"]].copy()
    functional_y = m1_df["is_functional_fraud"].astype(int).to_numpy()
    functional_scores = m1_df["functional_risk_score"].to_numpy()
    functional_pred = functional_scores >= threshold
    f_prec, f_rec, f_f1, f_tp, f_fp, f_fn, f_tn = precision_recall_f1(functional_y, functional_pred)
    functional_pr = pr_curve(functional_y, functional_scores)
    functional_recs = threshold_recommendations(
        functional_pr, precision_target, recall_target, len(m1_df), fraud_queue_budget
    )

    y_true = results_df["is_fraud"].astype(int).to_numpy()
    y_scores = results_df["functional_risk_score"].to_numpy()
    y_pred = y_scores >= threshold
    precision, recall, f1, tp, fp, fn, tn = precision_recall_f1(y_true, y_pred)
    pr_df = pr_curve(y_true, y_scores)
    recs = threshold_recommendations(
        pr_df, precision_target, recall_target, len(results_df), fraud_queue_budget
    )
    rules = rule_analytics(results_df, threshold)
    cohorts = error_cohorts(results_df, threshold)

    # print("\n" + "=" * 72)
    # print("MODEL 1 - FUNCTIONAL ANOMALY SCORING | SCOPE-ALIGNED REPORT")
    # print("=" * 72)
    # print(f"Dataset size : {len(results_df):,} profiles")
    # print(f"Fraud rate   : {y_true.mean() * 100:.1f}%")
    # print(f"M1 target    : functional + multi fraud vs legitimate ({len(m1_df):,} rows)")
    # print(f"Threshold    : {threshold:.2f}")
    # print(f"Throughput   : {results_df.attrs.get('throughput_per_sec', 0):,.0f} profiles/sec")
    # print(f"Latency      : mean {results_df['latency_ms'].mean():.3f} ms | p95 {results_df['latency_ms'].quantile(0.95):.3f} ms")
    # print(f"Extraction   : mean {results_df['extraction_ms'].mean():.3f} ms")
    # print(f"Scoring      : mean {results_df['scoring_ms'].mean():.3f} ms")
    # print(f"Fusion       : active rules mean {results_df['number_of_active_rules'].mean():.2f} | damping mean {results_df['damping_ratio'].mean():.3f}")

    print("\n" + "=" * 72)
    print("MODEL 1 — FUNCTIONAL CONSISTENCY REPORT")
    print("=" * 72)
    print(f"Dataset : {len(results_df):,} profiles  |  Fraud rate: {results_df['is_fraud'].mean()*100:.1f}%  |  Threshold: {threshold}")
    throughput = results_df.attrs.get("throughput_per_sec", 0)
    print(f"Speed   : {throughput:,.0f} profiles/sec  |  Latency: {results_df['latency_ms'].mean():.2f}ms mean")

    print("\nPrimary KPI - Functional Fraud Detection")
    print(f"  Functional AUC-ROC : {auc_roc(functional_y, functional_scores):.4f}")
    print(f"  Functional AUC-PR  : {auc_pr(functional_pr):.4f}")
    print(f"  Functional KS      : {target_ks_statistic(m1_df, 'is_functional_fraud'):.4f}")
    print(f"  Functional Cohen d : {cohen_d_scores(m1_df, 'is_functional_fraud'):.4f}")
    print(f"  Precision          : {f_prec:.4f} ({f_tp} functional TP, {f_fp} legitimate FP)")
    print(f"  Recall             : {f_rec:.4f} ({f_fn} missed functional/multi)")
    print(f"  F1 Score           : {f_f1:.4f}")
    print(f"  Confusion          : TN={f_tn}, FP={f_fp}, FN={f_fn}, TP={f_tp}")

    print("\nFunctional Threshold Recommendations")
    _print_threshold_row("Best functional F1", functional_recs["best_f1"])
    _print_threshold_row(f"Precision >= {precision_target:.2f}", functional_recs["precision_constrained"])
    _print_threshold_row(f"Recall >= {recall_target:.2f}", functional_recs["recall_constrained"])
    _print_threshold_row("Review budget", functional_recs["fraud_queue_budget"])

    # REMOVED - redundant with Recall@K
    # print("\nFunctional Recall@K / Queue Quality")
    # print(target_top_k(results_df, "is_functional_fraud").to_string(index=False, formatters=_formatters()))

    # print("\nFunctional Tail-Risk Concentration")
    # print(target_tail_risk_concentration(results_df, "is_functional_fraud").to_string(index=False, formatters=_formatters()))

    print("\nFunctional Calibration Buckets")
    print(target_calibration_buckets(m1_df, "is_functional_fraud")
          [["bucket", "count", "avg_score", "target_rate", "avg_confidence"]]
          .to_string(index=False, formatters=_formatters()))

    # REMOVED - redundant with Calibration Buckets
    # print("\nFunctional Score Histogram")
    # print(target_score_histogram(m1_df, "is_functional_fraud").to_string(index=False, formatters=_formatters()))

    # REMOVED - - internal statistical metric
    # print(f"\nFunctional Jensen-Shannon Divergence : {target_js_divergence_scores(m1_df, 'is_functional_fraud'):.4f}")

    # print("\nFunctional Percentile Overlap")
    # for key, value in target_percentile_overlap(m1_df, "is_functional_fraud").items():
    #     print(f"  {key:<42} {value:.4f}")

    # REMOVED - covered by Recall@K
    # print("\nFunctional Decile Lift (1 = highest-risk decile)")
    # print(target_decile_lift(results_df, "is_functional_fraud").head(10).to_string(index=False, formatters=_formatters()))

    print("\nFraud-Type Attribution Matrix")
    print(fraud_type_attribution_matrix(results_df, threshold).to_string(index=False, formatters=_formatters()))

    # REMOVED - broad label misleading for M1 (not designed to catch financial/ring fraud)
    # print("\nSecondary Context - Broad Fraud Label")
    # print(f"  AUC-ROC   : {auc_roc(y_true, y_scores):.4f}")
    # print(f"  AUC-PR    : {auc_pr(pr_df):.4f}")
    # print(f"  KS stat   : {ks_statistic(results_df):.4f}")
    # print(f"  Precision : {precision:.4f} ({tp} TP, {fp} FP)")
    # print(f"  Recall    : {recall:.4f} ({fn} missed fraud)")
    # print(f"  F1 Score  : {f1:.4f}")
    # print(f"  Confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # print("\nBroad-Label Threshold Recommendations")
    # _print_threshold_row("Best F1", recs["best_f1"])
    # _print_threshold_row(f"Precision >= {precision_target:.2f}", recs["precision_constrained"])
    # _print_threshold_row(f"Recall >= {recall_target:.2f}", recs["recall_constrained"])
    # _print_threshold_row("Fraud queue budget", recs["fraud_queue_budget"])

    # print("\nBroad-Label Threshold Sweep")
    # print(pr_df[pr_df["threshold"].isin([0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80])]
    #       [["threshold", "precision", "recall", "f1", "tp", "fp", "fn"]]
    #       .to_string(index=False, formatters=_formatters()))

    # print("\nCalibration Buckets")
    # print(calibration_buckets(results_df)[["bucket", "count", "avg_score", "fraud_rate", "avg_confidence"]]
    #       .to_string(index=False, formatters=_formatters()))

    # print("\nDecile Lift (1 = highest-risk decile)")
    # print(decile_lift(results_df).head(10).to_string(index=False, formatters=_formatters()))

    # print("\nScore Histogram")
    # print(score_histogram(results_df).to_string(index=False, formatters=_formatters()))

    # print("\nTop-K Precision")
    # print(top_k_precision(results_df).to_string(index=False, formatters=_formatters()))

    # print("\nTail-Risk Concentration")
    # print(tail_risk_concentration(results_df).to_string(index=False, formatters=_formatters()))

    # print(f"\nJensen-Shannon Divergence : {js_divergence_scores(results_df):.4f}")

    # print("\nPercentile Overlap")
    # for key, value in percentile_overlap(results_df).items():
    #     print(f"  {key:<36} {value:.4f}")

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
    cols = ["rule", "firing_rate", "fraud_firing_rate", "legit_firing_rate", "separation", "avg_effective_contribution", "fp_firing_rate"]
    print(rules["summary"][cols].head(15).to_string(index=False, formatters=_formatters()))
    # print(rules["summary"].head(15).to_string(index=False, formatters=_formatters()))

    # REMOVED - duplicates Rule Contribution
    # print("\nSignal Validity vs Functional Target")
    # print(rule_signal_validity(m1_df, "is_functional_fraud").head(20).to_string(index=False, formatters=_formatters()))

    # REMOVED - internal debugging metric
    # print("\nInterpretability Quality")
    # for key, value in interpretability_quality(results_df).items():
    #     print(f"  {key:<32} {value:.4f}")

    # REMOVED - debugging tool, not for regular runs
    # print("\nRule Abstention Report")
    # print(rules["abstentions"].head(20).to_string(index=False, formatters=_formatters()))

    # REMOVED - internal signal debugging
    # print("\nRule Pre-Threshold Distributions")
    # print(rules["distributions"].head(20).to_string(index=False, formatters=_formatters()))

    # REMOVED - covered by separation column in Rule Health
    # print("\nSignal Collapse Detection")
    # print(_empty_or_table(rules["collapse"].head(20), formatters=_formatters()))

    # REMOVED - duplicates Rule Contribution
    # print("\nInteraction Rule Analytics")
    # print(_empty_or_table(rules["interactions"].head(20), formatters=_formatters()))

    # REMOVED - internal fusion plumbing
    # print("\nContribution Saturation")
    # for key, value in fusion_saturation_report(results_df).items():
    #     print(f"  {key:<32} {value:.4f}")

    # REMOVED - only for speed optimization
    # print("\nRule Runtime Hotspots")
    # print(rule_runtime_hotspots(results_df).head(12).to_string(index=False, formatters=_formatters()))

    print("\nFalse-Positive-Heavy Rules")
    print(_empty_or_table(rules["fp_heavy"].head(10), formatters=_formatters()))

    print("\nLow-Information Rules")
    print(_empty_or_table(rules["low_info"].head(10), formatters=_formatters()))

    # REMOVED - always empty in current dataset
    # print("\nRedundant/Correlated Rule Pairs")
    # print(_empty_or_table(rules["correlated"].head(10), formatters=_formatters()))

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


def _numeric_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not pd.isna(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _clean_fraud_type(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "legitimate"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "legitimate"
    return text


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
            "abstain_rate", "raw_mean", "raw_p95", "normalized_mean",
            "normalized_p95", "confidence_mean", "confidence_p10",
            "score_variance", "avg_before_fusion", "avg_after_fusion",
            "fraud_share", "legit_share", "capture_rate", "min_score",
            "tail_pct", "interaction_lift", "fraud_rate_when_fired",
            "avg_contribution",
            "target_rate", "target_share", "negative_share", "precision_at_k",
            "recall_at_k", "recall_at_tail", "target_firing_rate",
            "negative_firing_rate", "odds_ratio", "mutual_info", "p90_score",
            "threshold_recall",
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model 1 functional consistency")
    parser.add_argument("--dataset", type=str, default="../output/profiles.csv")
    parser.add_argument("--threshold", type=float, default=0.45)  # Phase 1 optimal: 55.1% precision, 34.9% recall
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--precision-target", type=float, default=0.80)
    parser.add_argument("--recall-target", type=float, default=0.80)
    parser.add_argument("--fraud-queue-budget", type=int, default=500)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed calibration, rule, and error diagnostics",
    )
    args = parser.parse_args()

    profiles = load_profiles(args.dataset, limit=args.limit)
    results_df = run_scoring(profiles)

    # M1 is deterministic, so it has no fit stage. When generated temporal
    # splits are available, select its threshold on validation and report only
    # the future test split as final performance.
    dataset_dir = os.path.dirname(os.path.abspath(args.dataset))
    split_dir = os.path.join(dataset_dir, "splits")
    val_path = os.path.join(split_dir, "val.csv")
    test_path = os.path.join(split_dir, "test.csv")
    evaluation_scope = "full_dataset"
    if os.path.exists(val_path) and os.path.exists(test_path):
        val_ids = set(pd.read_csv(val_path)["profile_id"])
        test_ids = set(pd.read_csv(test_path)["profile_id"])
        val_results = results_df[results_df["profile_id"].isin(val_ids)]
        test_results = results_df[results_df["profile_id"].isin(test_ids)]
        if not val_results.empty and not test_results.empty:
            val_curve = pr_curve(val_results["is_functional_fraud"].astype(int), val_results["functional_risk_score"])
            args.threshold = float(
                val_curve.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]["threshold"]
            )
            results_df = test_results
            evaluation_scope = "temporal_test"
            print(
                f"Temporal evaluation: threshold selected on validation ({args.threshold:.2f}); "
                f"final metrics on test ({len(results_df):,} profiles)."
            )
    if evaluation_scope == "full_dataset":
        print("WARNING: temporal split files not found; metrics are descriptive full-dataset analysis.")
    report_fn = print_report if args.verbose else print_concise_report
    report_fn(
        results_df,
        args.threshold,
        args.precision_target,
        args.recall_target,
        args.fraud_queue_budget,
    )

    if args.save:
        results_df.to_csv(args.save, index=False)
        print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()

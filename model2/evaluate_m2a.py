"""Evaluation and report generation for Model 2A."""

from __future__ import annotations

import math
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def _binary_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _ks_stat(y_true, scores):
    y_true = np.asarray(y_true).astype(bool)
    s = np.asarray(scores)
    pos = np.sort(s[y_true])
    neg = np.sort(s[~y_true])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    values = np.sort(np.unique(s))
    return float(np.max(np.abs(np.searchsorted(pos, values, side="right") / len(pos) - np.searchsorted(neg, values, side="right") / len(neg))))


def _safe_auc_roc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def _threshold_sweep(df, target_col):
    rows = []
    y = df[target_col].astype(int).to_numpy()
    scores = df["text_risk"].to_numpy()
    for t in np.linspace(0.0, 1.0, 101):
        m = _binary_metrics(y, scores, t)
        m["threshold"] = round(float(t), 2)
        rows.append(m)
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0].to_dict()
    p80 = sweep[sweep["precision"] >= 0.80].sort_values(["recall", "f1"], ascending=False)
    r80 = sweep[sweep["recall"] >= 0.80].sort_values(["precision", "f1"], ascending=False)
    return sweep, best, (p80.iloc[0].to_dict() if not p80.empty else None), (r80.iloc[0].to_dict() if not r80.empty else None)


def _top_k(df, target_col, ks=(100, 500, 1000)):
    ranked = df.sort_values("text_risk", ascending=False)
    total = ranked[target_col].sum()
    rows = []
    for k in ks:
        sub = ranked.head(min(k, len(ranked)))
        rows.append(
            {
                "top_k": len(sub),
                "precision_at_k": float(sub[target_col].mean()) if len(sub) else 0.0,
                "targets_captured": int(sub[target_col].sum()),
                "recall_at_k": float(sub[target_col].sum() / total) if total else 0.0,
                "minimum_score": float(sub["text_risk"].min()) if len(sub) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _tail(df, target_col):
    rows = []
    total = df[target_col].sum()
    ranked = df.sort_values("text_risk", ascending=False)
    for p in (0.01, 0.02, 0.05, 0.10):
        n = max(1, int(len(df) * p))
        tail = ranked.head(n)
        captured = int(tail[target_col].sum())
        rows.append(
            {
                "tail_pct": p,
                "profiles_reviewed": n,
                "targets_captured": captured,
                "recall": float(captured / total) if total else 0.0,
                "precision": float(tail[target_col].mean()) if len(tail) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _calibration(df, target_col):
    buckets = pd.cut(df["text_risk"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
    return df.assign(bucket=buckets).groupby("bucket", observed=False).agg(
        count=("profile_id", "count"),
        average_score=("text_risk", "mean"),
        target_rate=(target_col, "mean"),
    ).reset_index()


def _score_histogram(df, target_col):
    bins = np.linspace(0.0, 1.0, 11)
    rows = []
    target = df[target_col].astype(bool)
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (df["text_risk"] >= lo) & (df["text_risk"] <= hi if i == 9 else df["text_risk"] < hi)
        rows.append(
            {
                "score_bin": f"{lo:.1f}-{hi:.1f}",
                "target_count": int((mask & target).sum()),
                "legitimate_count": int((mask & ~target).sum()),
            }
        )
    return pd.DataFrame(rows)


def _percentile_overlap(df, target_col):
    target = df[target_col].astype(bool)
    pos = df.loc[target, "text_risk"]
    neg = df.loc[~target, "text_risk"]
    if pos.empty or neg.empty:
        return {}
    neg_p90 = float(neg.quantile(0.90))
    return {
        "target_p10": float(pos.quantile(0.10)),
        "target_p50": float(pos.quantile(0.50)),
        "target_p90": float(pos.quantile(0.90)),
        "legitimate_p10": float(neg.quantile(0.10)),
        "legitimate_p50": float(neg.quantile(0.50)),
        "legitimate_p90": neg_p90,
        "target_share_leq_legitimate_p90": float((pos <= neg_p90).mean()),
    }


def _decile_lift(df, target_col):
    ranked = df.sort_values("text_risk", ascending=False).copy()
    ranked["decile"] = pd.qcut(np.arange(len(ranked)), 10, labels=False, duplicates="drop") + 1
    target = ranked[target_col].astype(bool)
    base = target.mean()
    total = int(target.sum())
    out = ranked.groupby("decile").agg(
        count=("profile_id", "count"),
        target_rate=(target_col, "mean"),
        targets=(target_col, "sum"),
    ).reset_index()
    out["lift"] = out["target_rate"] / base if base else 0.0
    out["cumulative_target_capture"] = out["targets"] / total if total else 0.0
    return out


def _fmt_threshold_row(row):
    if row is None:
        return "none"
    return f"{row['threshold']:.2f} (P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f})"


def _fraud_attribution(df, threshold):
    rows = []
    for fraud_type, group in df.groupby("fraud_type"):
        rows.append(
            {
                "fraud_type": fraud_type,
                "count": len(group),
                "avg_text_risk": float(group["text_risk"].mean()),
                "p90_text_risk": float(group["text_risk"].quantile(0.90)),
                "threshold_recall": float((group["text_risk"] >= threshold).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_text_risk", ascending=False)


def _error_cases(df, target_col, threshold):
    target = df[target_col].astype(bool)
    fp = df[(~target) & (df["text_risk"] >= threshold)].copy()
    fn = df[(target) & (df["text_risk"] < threshold)].copy()
    fp["error_type"] = "false_positive"
    fn["error_type"] = "false_negative"
    return fp, fn


def build_report(df: pd.DataFrame, threshold: float, output_dir: str, runtime: dict[str, float], exploratory: bool) -> str:
    primary = df["m2a_target"].astype(int)
    y = primary.to_numpy()
    scores = df["text_risk"].to_numpy()
    auc_roc = _safe_auc_roc(y, scores)
    auc_pr = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else 0.0
    ks = _ks_stat(y, scores)
    binary = _binary_metrics(y, scores, threshold)
    sweep, best, prec80, rec80 = _threshold_sweep(df, "m2a_target")
    topk = _top_k(df, "m2a_target")
    tail = _tail(df, "m2a_target")
    calib = _calibration(df, "m2a_target")
    hist = _score_histogram(df, "m2a_target")
    overlap = _percentile_overlap(df, "m2a_target")
    deciles = _decile_lift(df, "m2a_target")
    attr = _fraud_attribution(df, threshold)
    fp, fn = _error_cases(df, "m2a_target", threshold)

    lines = [
        "=" * 78,
        "MODEL 2A - TEXT ANOMALY DETECTION",
        "=" * 78,
        f"Dataset size: {len(df):,}",
        f"Primary target positives: {int(y.sum()):,}",
        f"Threshold used: {threshold:.2f}",
        f"Exploratory mode: {exploratory}",
        f"Interests column present: {'interests' in df.columns or 'partner_preferences' in df.columns}",
        "",
        "Primary Scope-Aligned Metrics",
        f"  AUC-ROC   : {auc_roc:.4f}",
        f"  AUC-PR    : {auc_pr:.4f}",
        f"  KS stat   : {ks:.4f}",
        f"  Precision : {binary['precision']:.4f}",
        f"  Recall    : {binary['recall']:.4f}",
        f"  F1        : {binary['f1']:.4f}",
        f"  Confusion : TN={binary['tn']}, FP={binary['fp']}, FN={binary['fn']}, TP={binary['tp']}",
        "",
        "Threshold Recommendations",
        f"  Best F1 threshold: {best['threshold']:.2f} (P={best['precision']:.4f}, R={best['recall']:.4f}, F1={best['f1']:.4f})",
        f"  Precision >= 0.80: {_fmt_threshold_row(prec80)}",
        f"  Recall >= 0.80: {_fmt_threshold_row(rec80)}",
        "",
        "Recall@K / Queue Quality",
        topk.to_string(index=False),
        "",
        "Tail-Risk Concentration",
        tail.to_string(index=False),
        "",
        "Calibration Buckets",
        calib.to_string(index=False),
        "",
        "Score Histogram",
        hist.to_string(index=False),
        "",
        "Percentile Overlap",
    ]
    for k, v in overlap.items():
        lines.append(f"  {k:<34} {v:.4f}")
    lines += [
        "",
        "Decile Lift",
        deciles.to_string(index=False),
        "",
        "Fraud-Type Attribution",
        attr.to_string(index=False),
        "",
        "False Positive Analysis",
        fp[["profile_id", "fraud_type", "max_similarity", "exact_combined_text_duplicate_count", "exact_bio_duplicate_count", "text_risk"]].head(20).to_string(index=False) if not fp.empty else "  none",
        "",
        "False Negative Analysis",
        fn[["profile_id", "fraud_type", "max_similarity", "exact_combined_text_duplicate_count", "exact_bio_duplicate_count", "text_risk"]].head(20).to_string(index=False) if not fn.empty else "  none",
        "",
        "Runtime",
        f"  nn_fit_time_sec   : {runtime.get('nn_fit_time_sec', 0.0):.4f}",
        f"  nn_query_time_sec : {runtime.get('nn_query_time_sec', 0.0):.4f}",
        f"  total_time_sec    : {runtime.get('total_time_sec', 0.0):.4f}",
    ]
    if exploratory:
        lines += ["", "Warnings / Leakage Notes", "  No temporal split files were found; TF-IDF was fit on the full dataset."]
    report = "\n".join(lines)
    with open(os.path.join(output_dir, "m2a_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report

"""Evaluation and reporting for Model 4 community trust scoring."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

SCORE_COL = "community_trust_risk"
TARGET_COL = "m4_target"


def _binary_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _safe_auc_roc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def _sweep_thresholds(df):
    y = df[TARGET_COL].astype(int).to_numpy()
    scores = df[SCORE_COL].to_numpy()
    rows = []
    for t in np.linspace(0.0, 1.0, 101):
        row = _binary_metrics(y, scores, t)
        row["threshold"] = round(float(t), 2)
        rows.append(row)
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0].to_dict()
    p80 = sweep[sweep["precision"] >= 0.80].sort_values(["recall", "f1"], ascending=False)
    r80 = sweep[sweep["recall"] >= 0.80].sort_values(["precision", "f1"], ascending=False)
    return sweep, best, (p80.iloc[0].to_dict() if not p80.empty else None), (r80.iloc[0].to_dict() if not r80.empty else None)


def _fmt_threshold_row(row):
    if row is None:
        return "none"
    return f"{row['threshold']:.2f} (P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f})"


def _top_k(df, ks=(100, 500, 1000)):
    ranked = df.sort_values(SCORE_COL, ascending=False)
    total = ranked[TARGET_COL].sum()
    rows = []
    for k in ks:
        sub = ranked.head(min(k, len(ranked)))
        rows.append({
            "top_k": len(sub),
            "precision_at_k": float(sub[TARGET_COL].mean()) if len(sub) else 0.0,
            "targets_captured": int(sub[TARGET_COL].sum()),
            "recall_at_k": float(sub[TARGET_COL].sum() / total) if total else 0.0,
            "minimum_trust_risk_score": float(sub[SCORE_COL].min()) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows)


def _tail(df):
    ranked = df.sort_values(SCORE_COL, ascending=False)
    total = ranked[TARGET_COL].sum()
    rows = []
    for pct in (0.01, 0.02, 0.05, 0.10):
        n = max(1, int(len(df) * pct))
        sub = ranked.head(n)
        captured = int(sub[TARGET_COL].sum())
        rows.append({
            "tail_pct": pct,
            "profiles_reviewed": n,
            "targets_captured": captured,
            "precision": float(sub[TARGET_COL].mean()) if len(sub) else 0.0,
            "recall": float(captured / total) if total else 0.0,
        })
    return pd.DataFrame(rows)


def _fraud_attribution(df, threshold):
    rows = []
    for fraud_type, group in df.groupby("fraud_type"):
        n_with_reports = int((group["n_reports"] > 0).sum())
        rows.append({
            "fraud_type": fraud_type,
            "count": len(group),
            "with_reports": n_with_reports,
            "coverage_pct": 100.0 * n_with_reports / len(group) if len(group) else 0.0,
            "mean_trust_risk": float(group[SCORE_COL].mean()),
            "p90_trust_risk": float(group[SCORE_COL].quantile(0.90)),
            "threshold_recall": float((group[SCORE_COL] >= threshold).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_trust_risk", ascending=False)


def _coverage_stats(df):
    reported = df["n_reports"] > 0
    fraud = df["is_fraud"].astype(bool)
    return {
        "total_profiles": len(df),
        "profiles_with_reports": int(reported.sum()),
        "fraud_profiles": int(fraud.sum()),
        "fraud_profiles_with_reports": int((fraud & reported).sum()),
        "fraud_coverage_pct": 100.0 * (fraud & reported).sum() / max(fraud.sum(), 1),
        "legit_profiles_with_reports": int((~fraud & reported).sum()),
        "legit_false_report_rate_pct": 100.0 * (~fraud & reported).sum() / max((~fraud).sum(), 1),
    }


def _error_cases(df, threshold):
    target = df[TARGET_COL].astype(bool)
    fp = df[(~target) & (df[SCORE_COL] >= threshold)].copy()
    fn = df[(target) & (df[SCORE_COL] < threshold)].copy()
    fp["error_type"] = "false_positive"
    fn["error_type"] = "false_negative"
    return fp, fn


def build_concise_report(df: pd.DataFrame, threshold: float, evaluation_df: pd.DataFrame | None = None, evaluation_scope: str = "full_dataset") -> str:
    evaluated = df if evaluation_df is None else evaluation_df
    target = evaluated[TARGET_COL].astype(int).to_numpy()
    scores = evaluated[SCORE_COL].to_numpy()
    metrics = _binary_metrics(target, scores, threshold)
    auc_roc = _safe_auc_roc(target, scores)
    auc_pr = float(average_precision_score(target, scores)) if len(np.unique(target)) > 1 else 0.0
    _, best, p80, r80 = _sweep_thresholds(evaluated)
    coverage = _coverage_stats(evaluated)

    lines = [
        "",
        "=" * 60,
        "MODEL 4 - COMMUNITY TRUST REPORT",
        "=" * 60,
        f"Dataset    : {len(df):,} profiles",
        f"Evaluation : {evaluation_scope} ({len(evaluated):,} profiles)",
        f"Threshold  : {threshold:.2f}",
        "",
        "Coverage (M4 only scores profiles that have received reports)",
        f"  Fraud profiles with >=1 report : {coverage['fraud_profiles_with_reports']:,}/{coverage['fraud_profiles']:,} ({coverage['fraud_coverage_pct']:.1f}%)",
        f"  Legit profiles with >=1 report : {coverage['legit_profiles_with_reports']:,} ({coverage['legit_false_report_rate_pct']:.1f}% false-report rate)",
        "",
        "Primary metrics",
        f"  AUC-ROC   : {auc_roc:.4f}",
        f"  AUC-PR    : {auc_pr:.4f}",
        f"  Precision : {metrics['precision']:.4f} ({metrics['tp']} TP, {metrics['fp']} FP)",
        f"  Recall    : {metrics['recall']:.4f} ({metrics['fn']} missed)",
        f"  F1 Score  : {metrics['f1']:.4f}",
        f"  Confusion : TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}",
        "",
        "Threshold recommendations",
        f"  Best F1          : {_fmt_threshold_row(best)}",
        f"  Precision >= 0.80: {_fmt_threshold_row(p80)}",
        f"  Recall >= 0.80   : {_fmt_threshold_row(r80)}",
        "",
        "Detailed diagnostics: m4_report.txt",
        "=" * 60,
    ]
    return "\n".join(lines)


def build_report(
    df: pd.DataFrame,
    threshold: float,
    output_dir: str,
    runtime: dict[str, float],
    exploratory: bool,
    evaluation_df: pd.DataFrame | None = None,
    evaluation_scope: str = "full_dataset",
) -> str:
    evaluated = evaluation_df if evaluation_df is not None else df
    y = evaluated[TARGET_COL].astype(int).to_numpy()
    scores = evaluated[SCORE_COL].to_numpy()
    auc_roc = _safe_auc_roc(y, scores)
    auc_pr = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else 0.0
    metrics = _binary_metrics(y, scores, threshold)
    sweep, best, p80, r80 = _sweep_thresholds(evaluated)
    topk = _top_k(evaluated)
    tail = _tail(evaluated)
    attr = _fraud_attribution(evaluated, threshold)
    coverage = _coverage_stats(evaluated)
    fp, fn = _error_cases(evaluated, threshold)

    lines = [
        "=" * 78,
        "MODEL 4 - COMMUNITY TRUST MODEL",
        "=" * 78,
        f"Dataset size: {len(df):,}",
        f"Evaluation scope: {evaluation_scope} ({len(evaluated):,} profiles)",
        f"Target positives: {int(y.sum()):,}",
        f"Threshold used: {threshold:.2f}",
        f"Exploratory mode: {exploratory}",
        "",
        "Coverage",
        f"  Fraud profiles with >=1 report : {coverage['fraud_profiles_with_reports']:,}/{coverage['fraud_profiles']:,} ({coverage['fraud_coverage_pct']:.1f}%)",
        f"  Legit profiles with >=1 report : {coverage['legit_profiles_with_reports']:,} ({coverage['legit_false_report_rate_pct']:.1f}% false-report rate)",
        "  Note: profiles with zero reports always score 0 by design; M4 is",
        "  complementary to M1/M2, not a replacement, on profiles with no interaction history yet.",
        "",
        "Primary Scope-Aligned Metrics",
        f"  ROC-AUC   : {auc_roc:.4f}",
        f"  PR-AUC    : {auc_pr:.4f}",
        f"  Precision : {metrics['precision']:.4f}",
        f"  Recall    : {metrics['recall']:.4f}",
        f"  F1        : {metrics['f1']:.4f}",
        f"  Confusion : TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}",
        f"  FPR       : {metrics['fp'] / max(metrics['fp'] + metrics['tn'], 1):.4f}",
        f"  FNR       : {metrics['fn'] / max(metrics['fn'] + metrics['tp'], 1):.4f}",
        "",
        "Threshold Recommendations",
        f"  Best F1 threshold: {best['threshold']:.2f} (P={best['precision']:.4f}, R={best['recall']:.4f}, F1={best['f1']:.4f})",
        f"  Precision >= 0.80: {('none' if p80 is None else f'{p80['threshold']:.2f} (P={p80['precision']:.4f}, R={p80['recall']:.4f})')}",
        f"  Recall >= 0.80: {('none' if r80 is None else f'{r80['threshold']:.2f} (P={r80['precision']:.4f}, R={r80['recall']:.4f})')}",
        "",
        "Recall@K",
        topk.to_string(index=False),
        "",
        "Tail-Risk Concentration",
        tail.to_string(index=False),
        "",
        "Fraud-Type Attribution",
        attr.to_string(index=False),
        "",
        "False Positive Analysis",
        fp[["profile_id", "fraud_type", SCORE_COL, "n_reports", "distinct_reporters", "n_coordinated_reports", "dominant_report_type"]].head(20).to_string(index=False) if not fp.empty else "  none",
        "",
        "False Negative Analysis",
        fn[["profile_id", "fraud_type", SCORE_COL, "n_reports", "distinct_reporters", "n_coordinated_reports", "dominant_report_type"]].head(20).to_string(index=False) if not fn.empty else "  none",
        "",
        "Runtime",
    ]
    for key, value in runtime.items():
        lines.append(f"  {key:<28} {value:.4f}")
    if exploratory:
        lines += ["", "Leakage / Dataset Warnings", "  No split files were available; thresholding is exploratory."]
    report = "\n".join(lines)
    with open(os.path.join(output_dir, "m4_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report

"""Evaluation and reporting for Model 2-B."""

from __future__ import annotations

import os

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


def _safe_auc_roc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5


def _ks(y_true, scores):
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores)
    pos = np.sort(s[y])
    neg = np.sort(s[~y])
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    values = np.sort(np.unique(s))
    return float(np.max(np.abs(np.searchsorted(pos, values, side="right") / len(pos) - np.searchsorted(neg, values, side="right") / len(neg))))


def _sweep_thresholds(df, target_col):
    y = df[target_col].astype(int).to_numpy()
    scores = df["behavior_risk"].to_numpy()
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


def _top_k(df, target_col, ks=(100, 500, 1000)):
    ranked = df.sort_values("behavior_risk", ascending=False)
    total = ranked[target_col].sum()
    rows = []
    for k in ks:
        sub = ranked.head(min(k, len(ranked)))
        rows.append({
            "top_k": len(sub),
            "precision_at_k": float(sub[target_col].mean()) if len(sub) else 0.0,
            "targets_captured": int(sub[target_col].sum()),
            "recall_at_k": float(sub[target_col].sum() / total) if total else 0.0,
            "minimum_behavior_risk_score": float(sub["behavior_risk"].min()) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows)


def _tail(df, target_col):
    ranked = df.sort_values("behavior_risk", ascending=False)
    total = ranked[target_col].sum()
    rows = []
    for pct in (0.01, 0.02, 0.05, 0.10):
        n = max(1, int(len(df) * pct))
        sub = ranked.head(n)
        captured = int(sub[target_col].sum())
        rows.append({
            "tail_pct": pct,
            "profiles_reviewed": n,
            "targets_captured": captured,
            "precision": float(sub[target_col].mean()) if len(sub) else 0.0,
            "recall": float(captured / total) if total else 0.0,
        })
    return pd.DataFrame(rows)


def _fraud_attribution(df, threshold):
    rows = []
    for fraud_type, group in df.groupby("fraud_type"):
        rows.append({
            "fraud_type": fraud_type,
            "count": len(group),
            "mean_behavior_risk": float(group["behavior_risk"].mean()),
            "median_behavior_risk": float(group["behavior_risk"].median()),
            "p90_behavior_risk": float(group["behavior_risk"].quantile(0.90)),
            "threshold_recall": float((group["behavior_risk"] >= threshold).mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_behavior_risk", ascending=False)


def _behavior_characteristics(df, target_col, threshold):
    high = df[df["behavior_risk"] >= threshold]
    normal = df[df["behavior_risk"] < threshold]
    cols = [
        "events_per_day", "edits_per_day", "uploads_per_day", "mean_inter_event_seconds",
        "burstiness", "night_activity_ratio", "location_switch_rate",
    ]
    rows = []
    for col in cols:
        rows.append({
            "feature": col,
            "high_risk_mean": float(high[col].mean()) if len(high) else 0.0,
            "normal_mean": float(normal[col].mean()) if len(normal) else 0.0,
            "diff": float(high[col].mean() - normal[col].mean()) if len(high) and len(normal) else 0.0,
        })
    return pd.DataFrame(rows)


def _error_cases(df, target_col, threshold):
    target = df[target_col].astype(bool)
    fp = df[(~target) & (df["behavior_risk"] >= threshold)].copy()
    fn = df[(target) & (df["behavior_risk"] < threshold)].copy()
    fp["error_type"] = "false_positive"
    fn["error_type"] = "false_negative"
    return fp, fn


def build_concise_report(
    df: pd.DataFrame,
    threshold: float,
    experiment: str,
    evaluation_df: pd.DataFrame | None = None,
    evaluation_scope: str = "full_dataset",
) -> str:
    """Build the short terminal report; full diagnostics remain in m2b_report.txt."""
    evaluated = df if evaluation_df is None else evaluation_df
    target = evaluated["m2b_target"].astype(int).to_numpy()
    scores = evaluated["behavior_risk"].to_numpy()
    metrics = _binary_metrics(target, scores, threshold)
    auc_roc = _safe_auc_roc(target, scores)
    auc_pr = float(average_precision_score(target, scores)) if len(np.unique(target)) > 1 else 0.0
    _, best, p80, r80 = _sweep_thresholds(evaluated, "m2b_target")

    lines = [
        "",
        "=" * 60,
        "MODEL 2B - BEHAVIORAL ANOMALY REPORT",
        "=" * 60,
        f"Dataset    : {len(df):,} profiles",
        f"Evaluation : {evaluation_scope} ({len(evaluated):,} profiles)",
        f"Experiment : {experiment}",
        f"Threshold  : {threshold:.2f}",
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
        f"  Best F1         : {_fmt_threshold_row(best)}",
        f"  Precision >= 0.80: {_fmt_threshold_row(p80)}",
        f"  Recall >= 0.80   : {_fmt_threshold_row(r80)}",
        "",
        "Recall by fraud type",
    ]

    fraud_rows = []
    for fraud_type, group in evaluated.groupby("fraud_type"):
        fraud_rows.append({
            "fraud_type": fraud_type,
            "count": len(group),
            "recall": float((group["behavior_risk"] >= threshold).mean()),
        })
    if fraud_rows:
        lines.append(pd.DataFrame(fraud_rows).sort_values("recall").to_string(index=False))
    else:
        lines.append("  none")
    lines += [
        "",
        "Detailed diagnostics: m2b_report.txt",
        "=" * 60,
    ]
    return "\n".join(lines)


def build_report(
    df: pd.DataFrame,
    threshold: float,
    output_dir: str,
    runtime: dict[str, float],
    experiment: str,
    exploratory: bool,
    evaluation_df: pd.DataFrame | None = None,
    evaluation_scope: str = "full_dataset",
) -> str:
    evaluated = evaluation_df if evaluation_df is not None else df
    y = evaluated["m2b_target"].astype(int).to_numpy()
    scores = evaluated["behavior_risk"].to_numpy()
    auc_roc = _safe_auc_roc(y, scores)
    auc_pr = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else 0.0
    ks = _ks(y, scores)
    metrics = _binary_metrics(y, scores, threshold)
    sweep, best, p80, r80 = _sweep_thresholds(evaluated, "m2b_target")
    topk = _top_k(evaluated, "m2b_target")
    tail = _tail(evaluated, "m2b_target")
    attr = _fraud_attribution(evaluated, threshold)
    behavior_chars = _behavior_characteristics(evaluated, "m2b_target", threshold)
    fp, fn = _error_cases(evaluated, "m2b_target", threshold)

    lines = [
        "=" * 78,
        "MODEL 2B - BEHAVIORAL ANOMALY DETECTION",
        "=" * 78,
        f"Dataset size: {len(df):,}",
        f"Evaluation scope: {evaluation_scope} ({len(evaluated):,} profiles)",
        f"Target positives: {int(y.sum()):,}",
        f"Experiment: {experiment}",
        f"Threshold used: {threshold:.2f}",
        f"Exploratory mode: {exploratory}",
        "",
        "Primary Scope-Aligned Metrics",
        f"  ROC-AUC   : {auc_roc:.4f}",
        f"  PR-AUC    : {auc_pr:.4f}",
        f"  KS        : {ks:.4f}",
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
        "Behavioral Characteristics of High-Risk Profiles",
        behavior_chars.to_string(index=False),
        "",
        "False Positive Analysis",
        fp[["profile_id", "fraud_type", "behavior_risk", "events_per_day", "burstiness", "night_activity_ratio", "location_switch_rate"]].head(20).to_string(index=False) if not fp.empty else "  none",
        "",
        "False Negative Analysis",
        fn[["profile_id", "fraud_type", "behavior_risk", "events_per_day", "burstiness", "night_activity_ratio", "location_switch_rate"]].head(20).to_string(index=False) if not fn.empty else "  none",
        "",
        "Runtime",
    ]
    for key, value in runtime.items():
        lines.append(f"  {key:<28} {value:.4f}")
    if exploratory:
        lines += ["", "Leakage / Dataset Warnings", "  No split files were available; thresholding is exploratory."]
    report = "\n".join(lines)
    with open(os.path.join(output_dir, "m2b_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report

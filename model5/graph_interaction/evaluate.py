"""Evaluation helpers for Model 5 — shared by the degree-only baseline now
and the GAT/fusion evaluation later."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score


def precision_at_recall(y_true: np.ndarray, scores: np.ndarray, recall_target: float) -> dict:
    """Precision at the lowest threshold that still achieves >= recall_target.

    This is the best-case precision for that recall level: as the threshold
    rises, recall falls monotonically while precision generally rises, so
    the point right at the recall_target crossing is the most favorable
    precision a operator get while still catching that fraction of fraud.
    """
    if y_true.sum() == 0:
        return {"recall_target": recall_target, "achieved_recall": 0.0, "precision": 0.0, "threshold": None}
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns recall descending from 1.0 at thresholds[0]
    # to 0 at the end; precision/recall have one more entry than thresholds.
    eligible = np.where(recall[:-1] >= recall_target)[0]
    if len(eligible) == 0:
        return {"recall_target": recall_target, "achieved_recall": float(recall[0]), "precision": float(precision[0]), "threshold": float(thresholds[0]) if len(thresholds) else None}
    idx = eligible[np.argmin(recall[:-1][eligible])]
    return {
        "recall_target": recall_target,
        "achieved_recall": float(recall[idx]),
        "precision": float(precision[idx]),
        "threshold": float(thresholds[idx]) if idx < len(thresholds) else None,
    }


def evaluate_group_vs_legit(
    group_scores: np.ndarray,
    legit_scores: np.ndarray,
    recall_targets: tuple[float, ...],
) -> dict:
    y = np.concatenate([np.ones(len(group_scores)), np.zeros(len(legit_scores))])
    s = np.concatenate([group_scores, legit_scores])
    row = {"n": len(group_scores)}
    try:
        row["roc_auc"] = float(roc_auc_score(y, s))
    except ValueError:
        row["roc_auc"] = float("nan")
    for target in recall_targets:
        r = precision_at_recall(y, s, target)
        row[f"precision_at_recall_{int(target*100)}"] = r["precision"]
        row[f"achieved_recall_{int(target*100)}"] = r["achieved_recall"]
    return row


def build_baseline_report(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    lines = [
        "=" * 78,
        "MODEL 5 — DEGREE-ONLY BASELINE (Step 3)",
        "=" * 78,
        "Features: log1p(in_degree), log1p(out_degree), log1p(degree_ratio) only.",
        "Precision reported at fixed recall (not just AUC) — with a large legit",
        "low-activity tail, 'few connections' alone flags many innocent users,",
        "so precision at an operating point shows the real cost.",
        "",
        df.to_string(index=False),
    ]
    return "\n".join(lines)

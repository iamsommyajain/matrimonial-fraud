"""Per-profile community trust aggregation for Model 4.

Reports are combined with noisy-OR fusion, the same style M1 uses to combine
rule contributions: risk = 1 - Π(1 - weight_i). This makes several
independent, credible reports compound strongly while a single low-weight
report barely moves the score, and the result is naturally bounded in [0, 1].

`cutoff_time` lets a caller ask "what would this profile's trust score have
been as of time T" using only reports filed by then — this is what makes the
score living/dynamic in production (it is recomputed as new reports arrive)
even though our evaluation in evaluate.py scores everything as of the latest
available report, matching M1/M2's static evaluation style.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model4.community_trust.config import SCORE_CAP
from model4.community_trust.features import compute_report_weights


@dataclass
class TrustScoreArtifacts:
    scores: pd.DataFrame
    weighted_reports: pd.DataFrame


def _noisy_or(weights: pd.Series) -> float:
    if weights.empty:
        return 0.0
    return float(1.0 - np.prod(1.0 - weights.to_numpy()))


def _dominant_report_type(group: pd.DataFrame) -> str:
    if group.empty:
        return ""
    return group.loc[group["report_weight_computed"].idxmax(), "report_type"]


def aggregate_trust_scores(
    reports_df: pd.DataFrame,
    profile_ids: pd.Series,
    cutoff_time: pd.Timestamp | None = None,
) -> TrustScoreArtifacts:
    weighted = compute_report_weights(reports_df)
    if cutoff_time is not None:
        weighted = weighted[weighted["timestamp"] <= cutoff_time]

    rows = []
    for reported_id, group in weighted.groupby("reported_profile_id"):
        raw_risk = _noisy_or(group["report_weight_computed"])
        rows.append({
            "profile_id": reported_id,
            "n_reports": len(group),
            "distinct_reporters": group["reporter_profile_id"].nunique(),
            "weighted_evidence_sum": float(group["report_weight_computed"].sum()),
            "max_report_weight": float(group["report_weight_computed"].max()),
            "n_coordinated_reports": int(group["is_coordinated_computed"].sum()),
            "dominant_report_type": _dominant_report_type(group),
            "community_trust_risk": min(raw_risk, SCORE_CAP),
        })
    agg = pd.DataFrame(rows)

    scaffold = pd.DataFrame({"profile_id": profile_ids})
    scores = scaffold.merge(agg, on="profile_id", how="left")
    fill_zero = ["n_reports", "distinct_reporters", "weighted_evidence_sum", "max_report_weight", "n_coordinated_reports", "community_trust_risk"]
    for col in fill_zero:
        scores[col] = scores[col].fillna(0)
    scores["dominant_report_type"] = scores["dominant_report_type"].fillna("")
    scores["n_reports"] = scores["n_reports"].astype(int)
    scores["distinct_reporters"] = scores["distinct_reporters"].astype(int)
    scores["n_coordinated_reports"] = scores["n_coordinated_reports"].astype(int)

    return TrustScoreArtifacts(scores, weighted)

"""Per-report weighting logic for Model 4.

Turns each raw report row into a single credibility-weighted evidence value
in [0, 1]. Nothing here reads the dataset's precomputed `report_weight` or
`is_coordinated_report` columns — both are recomputed from raw fields so the
weighting logic is owned by the model (see config.py for rationale).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model4.community_trust.config import (
    COORDINATION_CLUSTER_FRACTION,
    COORDINATION_DISCOUNT,
    COORDINATION_WINDOW_HOURS,
    DEFAULT_REPORT_TYPE_SEVERITY,
    REPORTER_AGE_NORMALIZATION_DAYS,
    REPORTER_FLAGGED_MULTIPLIER,
    REPORTER_VERIFIED_BONUS,
    REPORT_TYPE_SEVERITY,
    SINGLE_REPORT_WEIGHT_CAP,
)


def reporter_credibility(account_age_days: pd.Series, is_verified: pd.Series, is_flagged: pd.Series) -> pd.Series:
    """Credibility in [0, 1] from reporter tenure, verification, and flag status."""
    age_score = np.minimum(1.0, np.log1p(account_age_days.clip(lower=0)) / np.log1p(REPORTER_AGE_NORMALIZATION_DAYS))
    credibility = age_score + np.where(is_verified, REPORTER_VERIFIED_BONUS, 0.0)
    credibility = np.where(is_flagged, credibility * REPORTER_FLAGGED_MULTIPLIER, credibility)
    return pd.Series(np.clip(credibility, 0.0, 1.0), index=account_age_days.index)


def report_type_severity(report_type: pd.Series) -> pd.Series:
    return report_type.map(REPORT_TYPE_SEVERITY).fillna(DEFAULT_REPORT_TYPE_SEVERITY)


def _flag_coordinated(timestamps: pd.Series) -> pd.Series:
    """Flag reports that fall inside a window shared by more than half of a
    profile's reports — an organised-reporting (brigade) pattern rather than
    independent users noticing a problem separately.
    """
    if len(timestamps) <= 1:
        return pd.Series([False] * len(timestamps), index=timestamps.index)
    ts = timestamps.sort_values()
    window = pd.Timedelta(hours=COORDINATION_WINDOW_HOURS)
    values = ts.to_numpy()
    flagged = pd.Series(False, index=ts.index)
    for idx, t in ts.items():
        count_in_window = int(((values >= t) & (values <= t + window)).sum())
        if count_in_window > len(ts) * COORDINATION_CLUSTER_FRACTION:
            flagged.loc[idx] = True
    return flagged.reindex(timestamps.index)


def detect_coordinated_reports(reports_df: pd.DataFrame) -> pd.Series:
    return reports_df.groupby("reported_profile_id")["timestamp"].transform(
        lambda ts: _flag_coordinated(ts)
    ).astype(bool)


def compute_report_weights(reports_df: pd.DataFrame) -> pd.DataFrame:
    """Return reports_df with credibility, severity, coordination, and the
    final per-report evidence weight attached.
    """
    out = reports_df.copy()
    out["credibility"] = reporter_credibility(
        out["reporter_account_age_days"], out["reporter_is_verified"], out["reporter_is_flagged"]
    )
    out["severity"] = report_type_severity(out["report_type"])
    out["is_coordinated_computed"] = detect_coordinated_reports(out)
    out["coordination_discount"] = np.where(out["is_coordinated_computed"], COORDINATION_DISCOUNT, 1.0)
    out["report_weight_computed"] = (
        out["credibility"] * out["severity"] * out["coordination_discount"]
    ).clip(0.0, SINGLE_REPORT_WEIGHT_CAP)
    return out

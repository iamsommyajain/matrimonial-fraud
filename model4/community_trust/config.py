"""
Central configuration for Model 4 community trust scoring.

Every weight below is derived from RAW reporter/report fields at scoring time
(reporter_account_age_days, reporter_is_verified, reporter_is_flagged, report_type,
report timestamps). We deliberately never read the dataset's precomputed
`report_weight` / `is_coordinated_report` columns — those are generation-time
scaffolding, not something a live system would have; Model 4 recomputes its own
credibility and coordination signals so the logic lives in the model, same as
M1's rule weights and M2's feature engineering.

We also never use `resolution_status` / `outcome` as model inputs: the
generator assigns them once per profile based on how many reports it already
received, so they would leak the label.
"""

from __future__ import annotations

# ── Reporter credibility ────────────────────────────────────────────────────
# Account age matters most in the first year; a brand-new account contributes
# almost nothing, a verified year-plus account contributes close to full weight.
REPORTER_AGE_NORMALIZATION_DAYS = 365.0
REPORTER_VERIFIED_BONUS = 0.15
REPORTER_FLAGGED_MULTIPLIER = 0.05  # a report from a reporter who is themselves flagged is nearly worthless

# ── Report-type severity ────────────────────────────────────────────────────
# Not all report categories carry the same fraud signal for a matrimonial
# platform; financial/scam reports are the strongest signal, harassment the
# weakest (real, but less specific to profile fraud).
REPORT_TYPE_SEVERITY = {
    "financial_solicitation": 1.00,
    "scam": 1.00,
    "fake_identity": 0.80,
    "off_platform_pressure": 0.70,
    "scripted_conversation": 0.60,
    "harassment": 0.40,
}
DEFAULT_REPORT_TYPE_SEVERITY = 0.50

# ── Coordinated-reporting (brigade) detection ───────────────────────────────
# Recomputed independently from raw report timestamps per reported profile:
# if more than half of a profile's reports cluster inside a short window,
# those clustered reports look organised rather than independent, so their
# weight is discounted.
COORDINATION_WINDOW_HOURS = 2.0
COORDINATION_CLUSTER_FRACTION = 0.5
COORDINATION_DISCOUNT = 0.5

# ── Corroboration requirement ───────────────────────────────────────────────
# A single report, however credible the reporter, is still one person's
# account of an interaction and should not alone be enough to convict a
# profile (that would make the model trivially gameable by one false
# accusation). Capping the weight any single report can contribute means a
# profile only crosses into "high"/"critical" risk once multiple independent
# reports corroborate each other via the noisy-OR aggregation.
SINGLE_REPORT_WEIGHT_CAP = 0.50

# ── Aggregation ──────────────────────────────────────────────────────────────
# Reports are combined with the same noisy-OR fusion M1 uses for rule
# contributions: risk = 1 - Π(1 - weight_i). This makes independent reports
# compound strongly while saturating in [0, 1].
SCORE_CAP = 0.99

RISK_THRESHOLDS = {
    "low": (0.00, 0.30),
    "medium": (0.30, 0.55),
    "high": (0.55, 0.75),
    "critical": (0.75, 1.01),
}


def risk_level(score: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "critical"

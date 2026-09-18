"""
report_generator.py

Generates synthetic user reports for Model 4 (Community Trust Model).
Each report represents a user flagging a suspicious profile after interaction.

Schema:
    report_id               — unique ID
    reported_profile_id     — profile being reported
    reporter_profile_id     — profile filing the report
    report_type             — scam / fake_identity / harassment / financial_solicitation /
                              off_platform_pressure / scripted_conversation
    timestamp               — when the report was filed
    report_weight           — credibility score [0, 1] based on reporter attributes
    reporter_account_age_days
    reporter_is_verified    — has reporter completed basic verification
    reporter_is_flagged     — is the reporter themselves suspicious (reduces weight)
    resolution_status       — pending / confirmed / dismissed
    outcome                 — none / warned / suspended / banned
    is_coordinated_report   — True if reporter is closely connected to other reporters
                              of the same profile (brigade detection)

Design principles:
    - Fraud profiles get 3-10 reports, mostly from legitimate users
    - Legitimate profiles get 0-1 false reports (noise)
    - report_weight is computed from reporter attributes, not stored as a magic number
    - Financial scam and coordinated ring profiles attract the most reports
    - Reports trickle in over days/weeks after first contact (realistic timeline)
"""

import random
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPORT_TYPES = [
    "financial_solicitation",
    "fake_identity",
    "scripted_conversation",
    "off_platform_pressure",
    "harassment",
    "scam",
]

# Which fraud types attract which report categories (weighted)
FRAUD_TYPE_TO_REPORT_WEIGHTS = {
    "financial_scam": {
        "financial_solicitation": 0.40,
        "off_platform_pressure":  0.25,
        "scam":                   0.20,
        "scripted_conversation":  0.10,
        "fake_identity":          0.04,
        "harassment":             0.01,
    },
    "functional": {
        "fake_identity":          0.50,
        "scripted_conversation":  0.25,
        "scam":                   0.15,
        "financial_solicitation": 0.05,
        "off_platform_pressure":  0.03,
        "harassment":             0.02,
    },
    "template_bio": {
        "fake_identity":          0.45,
        "scripted_conversation":  0.35,
        "scam":                   0.10,
        "financial_solicitation": 0.05,
        "off_platform_pressure":  0.03,
        "harassment":             0.02,
    },
    "image_theft": {
        "fake_identity":          0.60,
        "scripted_conversation":  0.20,
        "scam":                   0.10,
        "financial_solicitation": 0.05,
        "off_platform_pressure":  0.03,
        "harassment":             0.02,
    },
    "coordinated_ring": {
        "scam":                   0.30,
        "financial_solicitation": 0.25,
        "fake_identity":          0.20,
        "off_platform_pressure":  0.15,
        "scripted_conversation":  0.08,
        "harassment":             0.02,
    },
    "multi": {
        "scam":                   0.25,
        "financial_solicitation": 0.25,
        "fake_identity":          0.20,
        "off_platform_pressure":  0.15,
        "scripted_conversation":  0.10,
        "harassment":             0.05,
    },
}

# Default for any fraud type not explicitly listed
DEFAULT_REPORT_WEIGHTS = {
    "fake_identity":          0.30,
    "scam":                   0.25,
    "scripted_conversation":  0.20,
    "financial_solicitation": 0.15,
    "off_platform_pressure":  0.07,
    "harassment":             0.03,
}

# Probability that a fraud profile gets reported AT ALL. Victims don't always
# notice, don't always bother reporting, or the platform never surfaces a
# reporting prompt to them. Without this, every fraud profile is guaranteed
# >=1 report, which makes "has a report" a near-perfect proxy for the label
# and defeats the point of a credibility-weighted trust model. Subtler fraud
# (template bios, generic functional inconsistency) gets reported less than
# fraud with a direct financial victim.
REPORT_PROBABILITY_BY_FRAUD_TYPE = {
    "financial_scam":    0.82,
    "coordinated_ring":  0.68,
    "multi":             0.72,
    "functional":        0.55,
    "template_bio":      0.45,
    "image_theft":       0.60,
}
DEFAULT_REPORT_PROBABILITY = 0.60

# Legitimate profiles that attract at least one false/noise report. A small
# slice of those get MULTIPLE false reports (jealous ex, rejected match,
# competitor) rather than exactly one — real overlap with the fraud
# distribution that a trust model has to be robust against, not an
# artificially clean 0-or-1 split.
LEGIT_FALSE_REPORT_RATE = 0.06
LEGIT_MULTI_REPORT_RATE = 0.15  # of those flagged, fraction that get 2-3 reports instead of 1

RESOLUTION_BY_N_REPORTS = {
    # (min_reports, max_reports) → (resolution, outcome) pool
    (1, 2):  [("pending", "none"), ("pending", "none"), ("dismissed", "none")],
    (3, 5):  [("pending", "none"), ("confirmed", "warned"), ("confirmed", "warned"), ("dismissed", "none")],
    (6, 10): [("confirmed", "warned"), ("confirmed", "suspended"), ("confirmed", "banned")],
}


# ─────────────────────────────────────────────────────────────────────────────
# Weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_report_weight(
    reporter_account_age_days: int,
    reporter_is_verified: bool,
    reporter_is_flagged: bool,
    is_coordinated_report: bool,
) -> float:
    """
    Compute a credibility weight for a single report.

    Rules (mirrors the design spec):
        - Verified long-standing user  → full weight (close to 1.0)
        - Brand new account            → very low weight
        - Reporter is itself flagged   → near-zero weight
        - Coordinated reports          → weight halved (brigade penalty)

    Returns a float in [0.0, 1.0].
    """
    # Base: account age contribution (logarithmic — first 30 days matter most)
    # Account age 0   → 0.0
    # Account age 30  → ~0.35
    # Account age 180 → ~0.70
    # Account age 365 → ~0.85
    age_score = min(1.0, np.log1p(reporter_account_age_days) / np.log1p(365))

    # Verification bonus
    verified_bonus = 0.15 if reporter_is_verified else 0.0

    weight = age_score + verified_bonus

    # Reporter is flagged — their report is nearly worthless
    if reporter_is_flagged:
        weight *= 0.05

    # Brigade penalty
    if is_coordinated_report:
        weight *= 0.5

    return round(min(1.0, max(0.0, weight)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Report type sampler
# ─────────────────────────────────────────────────────────────────────────────

def _sample_report_type(fraud_type: str) -> str:
    weights_map = FRAUD_TYPE_TO_REPORT_WEIGHTS.get(fraud_type, DEFAULT_REPORT_WEIGHTS)
    types  = list(weights_map.keys())
    probs  = list(weights_map.values())
    return random.choices(types, weights=probs, k=1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Resolution sampler
# ─────────────────────────────────────────────────────────────────────────────

def _sample_resolution(n_reports: int):
    for (lo, hi), pool in RESOLUTION_BY_N_REPORTS.items():
        if lo <= n_reports <= hi:
            return random.choice(pool)
    return random.choice([("confirmed", "banned"), ("confirmed", "suspended")])


# ─────────────────────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_reports_for_profile(
    profile: dict,
    legitimate_profile_ids: list,
) -> list:
    """
    Generate all reports for a single profile.

    Fraud profiles  → reported with a fraud-type-dependent probability
                       (not guaranteed); when reported, 1-10 reports depending
                       on fraud type, so early/weak-signal cases exist too.
    Legitimate      → ~6% get false/noise reports; most get 1, a minority get
                       2-3 (brigade-style false accusations).

    Returns a list of report dicts (may be empty).
    """
    is_fraud   = profile.get("is_fraud", False)
    fraud_type = profile.get("fraud_type") or "legitimate"
    created_at = datetime.fromisoformat(profile["created_at"])
    reported_id = profile["profile_id"]

    reports = []

    # ── Determine how many reports this profile attracts ─────────────────
    if not is_fraud:
        if random.random() > LEGIT_FALSE_REPORT_RATE:
            return []
        n_reports = random.randint(2, 3) if random.random() < LEGIT_MULTI_REPORT_RATE else 1
    else:
        # Not every fraud profile gets reported — victims don't always notice
        # or bother filing a report.
        report_probability = REPORT_PROBABILITY_BY_FRAUD_TYPE.get(fraud_type, DEFAULT_REPORT_PROBABILITY)
        if random.random() > report_probability:
            return []

        # Fraud type influences report volume; lower bound includes 1 so a
        # freshly-reported profile with only weak, early evidence exists too.
        base_counts = {
            "financial_scam":    (2, 10),
            "coordinated_ring":  (2, 9),
            "multi":             (2, 9),
            "functional":        (1, 6),
            "template_bio":      (1, 5),
            "image_theft":       (1, 5),
        }
        lo, hi = base_counts.get(fraud_type, (1, 6))
        n_reports = random.randint(lo, hi)

    # ── Determine resolution and outcome (same for all reports on profile) 
    resolution, outcome = _sample_resolution(n_reports)

    # ── Check for coordinated reporting (>3 reports = mild brigade risk) ──
    # We flag a report as coordinated if more than half the reports arrive
    # within a 2-hour window — sign of organised reporting or fake reports.
    report_timestamps = _generate_report_timestamps(created_at, n_reports, is_fraud)
    coordinated_flags = _detect_coordinated_timestamps(report_timestamps)

    # ── Sample reporters from legitimate pool ────────────────────────────
    # Exclude the reported profile itself
    available_reporters = [pid for pid in legitimate_profile_ids if pid != reported_id]
    if len(available_reporters) < n_reports:
        return []   # not enough users to simulate reports — skip
    reporters = random.sample(available_reporters, k=n_reports)

    # ── Build each report ────────────────────────────────────────────────
    for i, (reporter_id, ts, is_coord) in enumerate(
        zip(reporters, report_timestamps, coordinated_flags)
    ):
        account_age  = random.randint(1, 730)        # reporter's account age in days
        is_verified  = random.random() < 0.65        # 65% of reporters are verified
        is_flagged   = (not is_fraud) and (random.random() < 0.3)  # false reporters more likely flagged

        weight = compute_report_weight(
            reporter_account_age_days=account_age,
            reporter_is_verified=is_verified,
            reporter_is_flagged=is_flagged,
            is_coordinated_report=is_coord,
        )

        report_type = (
            _sample_report_type(fraud_type)
            if is_fraud
            else random.choice(["fake_identity", "harassment"])   # noise reports
        )

        reports.append({
            "report_id":                str(uuid4()),
            "reported_profile_id":      reported_id,
            "reporter_profile_id":      reporter_id,
            "report_type":              report_type,
            "timestamp":                ts.isoformat(),
            "report_weight":            weight,
            "reporter_account_age_days": account_age,
            "reporter_is_verified":     is_verified,
            "reporter_is_flagged":      is_flagged,
            "resolution_status":        resolution,
            "outcome":                  outcome,
            "is_coordinated_report":    is_coord,
            # Denormalised for quick model access
            "reported_is_fraud":        is_fraud,
            "reported_fraud_type":      fraud_type,
        })

    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_report_timestamps(
    profile_created_at: datetime,
    n_reports: int,
    is_fraud: bool,
) -> list:
    """
    Generate realistic report timestamps.

    Fraud profiles: reports trickle in 3-30 days after creation
                    (users need time to interact and realise something's wrong)
    Noise reports:  can come anytime — random within 60 days
    """
    timestamps = []
    for _ in range(n_reports):
        if is_fraud:
            # First contact happens ~3-10 days after creation
            # Reports come 1-20 days after first contact
            first_contact_offset = random.randint(3, 10)
            report_delay         = random.randint(1, 20)
            days_offset          = first_contact_offset + report_delay
        else:
            days_offset = random.randint(1, 60)

        hour   = random.randint(8, 23)
        minute = random.randint(0, 59)
        ts = profile_created_at + timedelta(days=days_offset, hours=hour, minutes=minute)
        timestamps.append(ts)

    return sorted(timestamps)


def _detect_coordinated_timestamps(timestamps: list, window_hours: float = 2.0) -> list:
    """
    Flag reports as coordinated if more than half fall within a 2-hour window.
    Returns a parallel bool list.
    """
    if len(timestamps) <= 1:
        return [False] * len(timestamps)

    window = timedelta(hours=window_hours)
    flagged = [False] * len(timestamps)

    # Sliding window count
    for i, ts in enumerate(timestamps):
        count_in_window = sum(
            1 for other in timestamps
            if ts <= other <= ts + window
        )
        if count_in_window > len(timestamps) / 2:
            flagged[i] = True

    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# Batch generator (called from generate_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_reports(all_profiles: list) -> list:
    """
    Generate reports for all profiles in the dataset.

    Args:
        all_profiles: list of profile dicts (as produced by generate_dataset.py)

    Returns:
        Flat list of report dicts ready to be saved as reports.csv
    """
    legitimate_ids = [
        p["profile_id"] for p in all_profiles if not p.get("is_fraud", False)
    ]

    all_reports = []
    for profile in all_profiles:
        reports = generate_reports_for_profile(profile, legitimate_ids)
        all_reports.extend(reports)

    return all_reports
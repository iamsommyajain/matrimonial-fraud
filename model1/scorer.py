"""
scorer.py

Takes extracted M1 features, calls all fuzzy rules, and combines
pair scores into a single weighted functional_risk_score ∈ [0, 1].

Weight rationale:
    age_experience      0.25  — hardest logical constraint, highest weight
    education_profession 0.20 — can't be a doctor without a medical degree
    salary_experience   0.20  — most common fabrication (inflated income)
    salary_profession   0.15  — cross-validates income vs role
    age_education       0.10  — PhD at 20 is obvious but less common
    profession_email    0.05  — soft signal, low weight
    education_salary    0.05  — cross-validates income vs qualification

Total: 1.00

Weights are defined as constants so you can tune them without
touching the logic.
"""

from fuzzy_rules import (
    score_age_experience,
    score_age_education,
    score_education_profession,
    score_profession_email,
    score_salary_experience,
    score_salary_profession,
    score_education_salary,
)

# ─────────────────────────────────────────────────────────────────────────────
# Weights — must sum to 1.0
# ─────────────────────────────────────────────────────────────────────────────

PAIR_WEIGHTS = {
    "age_experience":       0.25,
    "education_profession": 0.20,
    "salary_experience":    0.20,
    "salary_profession":    0.15,
    "age_education":        0.10,
    "profession_email":     0.05,
    "education_salary":     0.05,
}

# Flag threshold: if an individual pair score crosses this, add to flags list
PAIR_FLAG_THRESHOLD = 0.60

# Risk level thresholds for the final score
RISK_THRESHOLDS = {
    "low":      (0.00, 0.30),
    "medium":   (0.30, 0.55),
    "high":     (0.55, 0.75),
    "critical": (0.75, 1.01),
}


# ─────────────────────────────────────────────────────────────────────────────
# Pair score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pair_scores(features: dict) -> dict:
    """
    Run all fuzzy rules and return a dict of pair_name → score.
    If a required feature is missing, that pair returns 0.0 (no penalty).

    Args:
        features: output of feature_extractor.extract_m1_features()

    Returns:
        {
          "age_experience":       float,
          "age_education":        float,
          "education_profession": float,
          "profession_email":     float,
          "salary_experience":    float,
          "salary_profession":    float,
          "education_salary":     float,
        }
    """
    age       = features.get("age")
    edu       = features.get("education_level")
    exp       = features.get("years_experience")
    prof      = features.get("profession")
    salary    = features.get("annual_income_lpa")
    email_dom = features.get("email_domain")

    scores = {}

    # Age ↔ Experience
    scores["age_experience"] = (
        score_age_experience(age, exp)
        if age is not None and exp is not None
        else 0.0
    )

    # Age ↔ Education
    scores["age_education"] = (
        score_age_education(age, edu)
        if age is not None and edu is not None
        else 0.0
    )

    # Education ↔ Profession
    scores["education_profession"] = (
        score_education_profession(edu, prof)
        if edu is not None and prof is not None
        else 0.0
    )

    # Profession ↔ Email
    scores["profession_email"] = (
        score_profession_email(prof, email_dom)
        if prof is not None and email_dom is not None
        else 0.0
    )

    # Salary ↔ Experience
    scores["salary_experience"] = (
        score_salary_experience(salary, exp)
        if salary is not None and exp is not None
        else 0.0
    )

    # Salary ↔ Profession
    scores["salary_profession"] = (
        score_salary_profession(salary, prof)
        if salary is not None and prof is not None
        else 0.0
    )

    # Education ↔ Salary
    scores["education_salary"] = (
        score_education_salary(edu, salary)
        if edu is not None and salary is not None
        else 0.0
    )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Weighted fusion
# ─────────────────────────────────────────────────────────────────────────────

def compute_functional_risk_score(features: dict) -> tuple:
    """
    Compute the final functional risk score and all supporting outputs.

    Returns:
        (functional_risk_score, pair_scores, flags, risk_level)

        functional_risk_score: float ∈ [0, 1]   — 0=clean, 1=fraud
        pair_scores:           dict              — individual pair scores
        flags:                 list[str]         — pairs above threshold
        risk_level:            str               — low/medium/high/critical
    """
    pair_scores = compute_pair_scores(features)

    # Weighted sum — pairs with missing data already returned 0.0
    total_weight_used = 0.0
    weighted_sum      = 0.0

    for pair, score in pair_scores.items():
        w = PAIR_WEIGHTS.get(pair, 0.0)
        weighted_sum      += score * w
        total_weight_used += w

    # Normalise in case some weights were 0 (shouldn't happen but safe)
    if total_weight_used > 0:
        functional_risk_score = weighted_sum / total_weight_used
    else:
        functional_risk_score = 0.0

    functional_risk_score = round(min(1.0, max(0.0, functional_risk_score)), 4)

    # Flags: individual pairs that exceed the threshold
    flags = [
        pair for pair, score in pair_scores.items()
        if score >= PAIR_FLAG_THRESHOLD
    ]

    # Risk level
    risk_level = _classify_risk(functional_risk_score)

    return functional_risk_score, pair_scores, flags, risk_level


# ─────────────────────────────────────────────────────────────────────────────
# Risk level classifier
# ─────────────────────────────────────────────────────────────────────────────

def _classify_risk(score: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "critical"

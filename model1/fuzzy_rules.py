"""
Deterministic M1 rule registry.

Rule output semantics are strict:
  score: severity of the inconsistency/fraud signal.
  confidence: reliability/completeness of the evidence available to this rule.
  evidence_weight: global discriminatory value of this rule.

Rules are isolated: they consume feature subsets, do not call each other, and
never mutate shared state.
"""

from __future__ import annotations

from config import (
    CLUSTER_INCOME_RANGE,
    CLUSTER_MIN_EDU_RANK,
    CRITICAL_FIELD_PRIORS,
    EDU_INCOME_RANGE,
    EDU_MIN_COMPLETION_AGE,
    FEATURE_CRITICALITY,
    HIGH_CONCERN_CLUSTERS,
    MIN_START_AGE,
    PERSONAL_EMAIL_DOMAINS,
    RULE_EVIDENCE_WEIGHTS,
    RULE_WEIGHTS,
    expected_salary_range_by_experience,
)
from m1_types import RuleResult, RuleSpec, clamp01


def _linear_ramp(value: float, ok_below: float, max_penalty_at: float,
                 floor: float = 0.0, ceiling: float = 1.0) -> float:
    if value <= ok_below:
        return floor
    if value >= max_penalty_at:
        return ceiling
    ratio = (value - ok_below) / (max_penalty_at - ok_below)
    return floor + ratio * (ceiling - floor)


def _available_confidence(*values) -> float:
    observed = sum(1 for v in values if v is not None and v != "")
    return observed / len(values) if values else 0.0


def _missing_fields(named_values: dict) -> list[str]:
    return [name for name, value in named_values.items() if value is None or value == ""]


def _abstain_reason(score: float, confidence: float, missing_fields: list[str],
                    raw_signal=None, threshold=None) -> str | None:
    if missing_fields:
        return "missing_inputs"
    if confidence < 0.35:
        return "low_confidence"
    if raw_signal is None and threshold is not None:
        return "normalization_failure"
    if score <= 0.0:
        return "threshold_not_met"
    return None


def _rr(rule: str, score: float, confidence: float, reason: str,
        features: dict | None = None, missingness_impact: float = 0.0,
        raw_signal_value=None, normalized_signal_value: float | None = None,
        threshold_used=None, missing_fields: list[str] | None = None,
        abstained_reason: str | None = None,
        missingness_penalty: float = 0.0) -> RuleResult:
    score = clamp01(score)
    confidence = clamp01(confidence)
    missing_fields = missing_fields or []
    if normalized_signal_value is None and isinstance(raw_signal_value, (int, float)):
        normalized_signal_value = score
    if abstained_reason is None:
        abstained_reason = _abstain_reason(
            score, confidence, missing_fields, raw_signal_value, threshold_used
        )
    return RuleResult(
        rule=rule,
        score=score,
        confidence=confidence,
        evidence_weight=RULE_EVIDENCE_WEIGHTS[rule],
        reason=reason,
        features=features or {},
        missingness_impact=clamp01(missingness_impact),
        fired=score > 0.0 and abstained_reason is None,
        abstained_reason=abstained_reason,
        raw_signal_value=raw_signal_value,
        normalized_signal_value=normalized_signal_value,
        threshold_used=threshold_used,
        missing_fields=missing_fields,
        missingness_penalty=clamp01(missingness_penalty),
    )


def rule_age_experience(features: dict) -> RuleResult:
    d = features.get("demographic", {})
    c = features.get("career", {})
    derived = features.get("derived", {})
    age = d.get("age")
    years_exp = c.get("years_experience")
    cluster = c.get("profession_cluster") or "other"
    start_age = derived.get("career_start_age")
    min_start = MIN_START_AGE.get(cluster, 18)
    conf = _available_confidence(age, years_exp)
    missing = _missing_fields({"age": age, "years_experience": years_exp})

    if conf < 1.0:
        return _rr("age_experience", 0.0, conf, "Age or experience missing",
                   {"age": age, "years_experience": years_exp}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    if years_exp <= 0:
        return _rr("age_experience", 0.0, 0.95, "Zero experience is plausible",
                   {"age": age, "years_experience": years_exp},
                   raw_signal_value=0, threshold_used=min_start)
    if start_age is None or start_age >= min_start:
        return _rr("age_experience", 0.0, 0.95, "Experience is plausible for age",
                   {"age": age, "years_experience": years_exp, "career_start_age": start_age},
                   raw_signal_value=max(0, min_start - start_age), threshold_used=0)

    shortfall = min_start - start_age
    score = _linear_ramp(shortfall, 0.0, 6.0)
    return _rr("age_experience", score, 0.96,
               f"Career start age {start_age} is below expected minimum {min_start}",
               {"age": age, "years_experience": years_exp,
                "career_start_age": start_age, "expected_min_start_age": min_start},
               raw_signal_value=shortfall, normalized_signal_value=score,
               threshold_used=0)


def rule_age_education(features: dict) -> RuleResult:
    d = features.get("demographic", {})
    c = features.get("career", {})
    age = d.get("age")
    edu = c.get("education_level_normalized")
    min_age = EDU_MIN_COMPLETION_AGE.get(edu)
    conf = _available_confidence(age, edu) * (1.0 if min_age is not None else 0.65)
    missing = _missing_fields({"age": age, "education_level": edu})
    if age is None or not edu:
        return _rr("age_education", 0.0, conf, "Age or education missing",
                   {"age": age, "education_level": edu}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    if min_age is None or age >= min_age:
        return _rr("age_education", 0.0, conf, "Education is plausible for age",
                   {"age": age, "education_level": edu, "min_completion_age": min_age},
                   raw_signal_value=0 if min_age is not None else None, threshold_used=min_age)

    shortfall = min_age - age
    return _rr("age_education", _linear_ramp(shortfall, 0.0, 5.0), conf,
               f"{edu} is unusually early at age {age}",
               {"age": age, "education_level": edu, "min_completion_age": min_age},
               raw_signal_value=shortfall, threshold_used=0)


def rule_education_profession(features: dict) -> RuleResult:
    c = features.get("career", {})
    edu_rank = c.get("education_rank")
    edu = c.get("education_level_normalized")
    cluster = c.get("profession_cluster") or "other"
    profession = c.get("profession_normalized")
    min_rank = CLUSTER_MIN_EDU_RANK.get(cluster)
    conf = _available_confidence(edu_rank, profession) * (1.0 if min_rank else 0.55)
    missing = _missing_fields({"education_rank": edu_rank, "profession": profession})
    if edu_rank is None or not profession:
        return _rr("education_profession", 0.0, conf, "Education or profession missing",
                   {"education_level": edu, "profession": profession}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    if min_rank is None or edu_rank >= min_rank:
        return _rr("education_profession", 0.0, conf,
                   "Education is compatible with profession cluster",
                   {"education_rank": edu_rank, "profession_cluster": cluster,
                    "required_min_rank": min_rank},
                   raw_signal_value=0 if min_rank else None, threshold_used=min_rank)

    gap = min_rank - edu_rank
    score = _linear_ramp(gap, 0.0, 2.0, ceiling=0.9)
    return _rr("education_profession", score, conf,
               f"{profession} usually requires higher education than {edu}",
               {"education_rank": edu_rank, "profession_cluster": cluster,
                "required_min_rank": min_rank},
               raw_signal_value=gap, normalized_signal_value=score,
               threshold_used=0)


def rule_profession_email(features: dict) -> RuleResult:
    c = features.get("career", {})
    e = features.get("email", {})
    cluster = c.get("profession_cluster") or "other"
    profession = c.get("profession_normalized")
    domain = e.get("email_domain")
    conf = _available_confidence(profession, domain)
    missing = _missing_fields({"profession": profession, "email_domain": domain})
    if not domain or not profession:
        return _rr("profession_email", 0.0, conf, "Profession or email missing",
                   {"profession": profession, "email_domain": domain}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    if e.get("disposable_domain"):
        return _rr("profession_email", 0.72, 0.98, "Disposable email domain",
                   {"profession_cluster": cluster, "email_domain": domain},
                   raw_signal_value=1, normalized_signal_value=0.72,
                   threshold_used="disposable_domain")
    if cluster in HIGH_CONCERN_CLUSTERS and domain in PERSONAL_EMAIL_DOMAINS:
        return _rr("profession_email", 0.22, 0.88,
                   "High-verification profession uses personal email domain",
                   {"profession_cluster": cluster, "email_domain": domain},
                   raw_signal_value=1, normalized_signal_value=0.22,
                   threshold_used="high_concern_personal_domain")
    return _rr("profession_email", 0.0, conf, "Email domain is not profession-inconsistent",
               {"profession_cluster": cluster, "email_domain": domain},
               raw_signal_value=0, threshold_used="profession_email_mismatch")


def rule_salary_experience(features: dict) -> RuleResult:
    c = features.get("career", {})
    income = c.get("annual_income_lpa")
    years_exp = c.get("years_experience")
    cluster = c.get("profession_cluster") or "other"
    conf = _available_confidence(income, years_exp)
    missing = _missing_fields({"annual_income_lpa": income, "years_experience": years_exp})
    if income is None or years_exp is None:
        return _rr("salary_experience", 0.0, conf, "Income or experience missing",
                   {"income": income, "years_experience": years_exp}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    if cluster in {"business_entrepreneur", "student"}:
        conf *= 0.72

    soft_min, soft_max = expected_salary_range_by_experience(years_exp)
    score = 0.0
    direction = "within expected range"
    if income > soft_max:
        overshoot_ratio = (income - soft_max) / soft_max
        # Phase 1: Increased threshold from 0.10 to 0.40 to reduce false positives
        # Only flag if income is 40%+ above expected range (not just 10%)
        score = _linear_ramp(overshoot_ratio, 0.40, 1.50)
        direction = "above expected range"
    elif income < soft_min and cluster not in {"student", "business_entrepreneur"}:
        undershoot_ratio = (soft_min - income) / soft_min
        # Phase 1: Increased threshold from 0.15 to 0.40 to reduce false positives
        # Only flag if income is 40%+ below expected range (not just 15%)
        score = _linear_ramp(undershoot_ratio, 0.40, 1.00) * 0.35
        direction = "below expected range"

    return _rr("salary_experience", score, conf,
               f"Income is {direction} for experience",
               {"income": income, "years_experience": years_exp,
                "expected_min": soft_min, "expected_max": soft_max},
               raw_signal_value=(
                   (income - soft_max) / soft_max if income > soft_max
                   else (soft_min - income) / soft_min if income < soft_min else 0.0
               ),
               normalized_signal_value=score, threshold_used=0.40)


def rule_salary_profession(features: dict) -> RuleResult:
    c = features.get("career", {})
    derived = features.get("derived", {})
    income = c.get("annual_income_lpa")
    cluster = c.get("profession_cluster") or "other"
    band = derived.get("expected_income_band")
    ratio = derived.get("income_deviation_ratio")
    rng = CLUSTER_INCOME_RANGE.get(cluster, CLUSTER_INCOME_RANGE["other"])
    conf = _available_confidence(income, cluster) * (0.65 if cluster == "other" else 1.0)
    if income is None:
        return _rr("salary_profession", 0.0, conf, "Income missing",
                   {"income": income, "profession_cluster": cluster}, 1.0 - conf,
                   missing_fields=["annual_income_lpa"], missingness_penalty=1.0 - conf)
    if band == "above_expected":
        score = _linear_ramp((income - rng[1]) / max(rng[1], 1.0), 0.05, 1.2)
    elif band == "below_expected" and cluster not in {"student", "business_entrepreneur"}:
        score = _linear_ramp((rng[0] - income) / max(rng[0], 1.0), 0.10, 1.0) * 0.32
    else:
        score = 0.0
    return _rr("salary_profession", score, conf,
               f"Income band is {band or 'unknown'} for profession cluster",
               {"income": income, "profession_cluster": cluster, "expected_range": rng,
                "income_deviation_ratio": ratio},
               raw_signal_value=ratio, normalized_signal_value=score,
               threshold_used="profession_income_range")


def rule_education_salary(features: dict) -> RuleResult:
    c = features.get("career", {})
    income = c.get("annual_income_lpa")
    edu = c.get("education_level_normalized")
    rng = EDU_INCOME_RANGE.get(edu)
    conf = _available_confidence(income, edu) * (1.0 if rng else 0.55)
    missing = _missing_fields({"annual_income_lpa": income, "education_level": edu})
    if income is None or not edu:
        return _rr("education_salary", 0.0, conf, "Income or education missing",
                   {"income": income, "education_level": edu}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    if not rng:
        return _rr("education_salary", 0.0, conf, "Education level has no salary prior",
                   {"income": income, "education_level": edu},
                   abstained_reason="invalid_feature_combo")
    low, high = rng
    if income > high:
        score = _linear_ramp((income - high) / max(high, 1.0), 0.05, 1.2)
        reason = "Income is high for education level"
    elif income < low:
        score = _linear_ramp((low - income) / max(low, 1.0), 0.15, 1.0) * 0.22
        reason = "Income is low for education level"
    else:
        score = 0.0
        reason = "Income is compatible with education level"
    return _rr("education_salary", score, conf, reason,
               {"income": income, "education_level": edu, "expected_range": rng},
               raw_signal_value=(
                   (income - high) / max(high, 1.0) if income > high
                   else (low - income) / max(low, 1.0) if income < low else 0.0
               ),
               normalized_signal_value=score, threshold_used=0.05)


def rule_company_income(features: dict) -> RuleResult:
    c = features.get("career", {})
    derived = features.get("derived", {})
    income = c.get("annual_income_lpa")
    tier = c.get("company_tier_numeric")
    consistent = derived.get("company_income_consistent")
    conf = _available_confidence(income, tier)
    if consistent is None:
        return _rr("company_income", 0.0, conf, "Company tier or income unavailable",
                   {"income": income, "company_tier": tier}, 1.0 - conf,
                   missing_fields=_missing_fields({"annual_income_lpa": income, "company_tier": tier}),
                   missingness_penalty=1.0 - conf)
    score = 0.62 if consistent is False else 0.0
    return _rr("company_income", score, conf,
               "Company tier and income are inconsistent" if score else "Company tier and income align",
               {"income": income, "company_tier": tier},
               raw_signal_value=int(consistent is False), normalized_signal_value=score,
               threshold_used="company_income_consistent_false")


def rule_email_quality(features: dict) -> RuleResult:
    e = features.get("email", {})
    domain = e.get("email_domain")
    conf = 0.95 if domain else 0.0
    score = 0.0
    reasons = []
    if not domain:
        return _rr("email_quality", 0.0, conf, "Email missing",
                   {"email_domain": domain}, 1.0,
                   missing_fields=["email"], missingness_penalty=1.0)
    if e.get("disposable_domain"):
        score = max(score, 0.78)
        reasons.append("disposable domain")
    if e.get("typo_domain"):
        score = max(score, 0.58)
        reasons.append("typo-squatted domain")
    if e.get("excessive_digits"):
        score = max(score, 0.30)
        reasons.append("many username digits")
    if e.get("suspicious_keywords_present"):
        score = max(score, 0.28)
        reasons.append("suspicious username keyword")
    entropy = e.get("username_entropy")
    digit_ratio = e.get("digit_ratio")
    return _rr("email_quality", score, conf,
               "Email has " + ", ".join(reasons) if reasons else "Email quality signals are normal",
               {"email_domain": domain, "username_entropy": entropy,
                "digit_ratio": digit_ratio},
               raw_signal_value=len(reasons), normalized_signal_value=score,
               threshold_used="email_quality_flags")


def rule_behavioral_integrity(features: dict) -> RuleResult:
    b = features.get("behavioral", {})
    account_age = b.get("account_age_days")
    edit_count = b.get("profile_edit_count")
    ip_div = b.get("ip_diversity")
    edit_freq = b.get("edit_frequency")
    total_logins = b.get("n_total_logins")
    conf = _available_confidence(account_age, edit_count, total_logins)
    score = 0.0
    reasons = []
    if account_age is not None and account_age <= 2 and edit_count and edit_count >= 6:
        score = max(score, 0.55)
        reasons.append("many edits on a new account")
    if edit_freq is not None and edit_freq > 3.0:
        score = max(score, 0.45)
        reasons.append("high edit frequency")
    if (
        ip_div is not None
        and total_logins
        and total_logins >= 8
        and ip_div > 0.85
        and account_age is not None
        and account_age <= 30
    ):
        score = max(score, 0.28)
        reasons.append("high IP diversity on a new account")
    return _rr("behavioral_integrity", score, conf,
               ", ".join(reasons) if reasons else "Behavioral signals are not unusual",
               {"account_age_days": account_age, "profile_edit_count": edit_count,
                "edit_frequency": edit_freq, "ip_diversity": ip_div,
                "n_total_logins": total_logins},
               raw_signal_value=len(reasons), normalized_signal_value=score,
               threshold_used="behavioral_velocity")


def rule_textual_quality(features: dict) -> RuleResult:
    t = features.get("textual", {})
    word_count = t.get("bio_word_count")
    generic_hits = t.get("generic_language_hits") or 0
    repetition = t.get("bio_ngram_repetition_score") or 0.0
    conf = 0.75 if word_count is not None else 0.0
    score = 0.0
    if word_count is not None and word_count < 8:
        score = max(score, 0.28)
    if generic_hits >= 3:
        score = max(score, min(0.45, generic_hits / 8.0))
    if repetition > 0.20:
        score = max(score, min(0.50, repetition))
    return _rr("textual_quality", score, conf,
               "Text profile is sparse/generic/repetitive" if score else "Text profile is not unusually generic",
               {"bio_word_count": word_count, "generic_language_hits": generic_hits,
                "bio_ngram_repetition_score": repetition},
               raw_signal_value=max(generic_hits / 8.0, repetition, 1.0 if word_count is not None and word_count < 8 else 0.0),
               normalized_signal_value=score, threshold_used="text_quality_flags")


def rule_contextual_missingness(features: dict) -> RuleResult:
    c = features.get("career", {})
    b = features.get("behavioral", {})
    m = features.get("missingness", {})
    cluster = c.get("profession_cluster") or "other"
    missing = {
        field: bool(m.get(f"missing_{field}"))
        for field in CRITICAL_FIELD_PRIORS
    }
    critical_missing = [f for f in FEATURE_CRITICALITY["critical"] if missing.get(f)]
    suspicious_score = sum(CRITICAL_FIELD_PRIORS[field] for field in critical_missing)
    uncertainty_score = min(1.0, 0.12 * len(critical_missing))
    neutral_missing = []

    if cluster == "student" and missing.get("annual_income_lpa"):
        suspicious_score -= CRITICAL_FIELD_PRIORS["annual_income_lpa"] * 0.75
        neutral_missing.append("annual_income_lpa")
    if cluster == "business_entrepreneur" and missing.get("annual_income_lpa"):
        suspicious_score -= CRITICAL_FIELD_PRIORS["annual_income_lpa"] * 0.45

    core_sparse = missing.get("profession") and missing.get("education_level") and missing.get("annual_income_lpa")
    activity_sparse = (b.get("n_total_logins") in (None, 0)) and (b.get("profile_edit_count") in (None, 0))
    if core_sparse:
        suspicious_score += 0.18
    if core_sparse and activity_sparse:
        suspicious_score += 0.16
    if missing.get("email"):
        suspicious_score += 0.08

    score = clamp01(suspicious_score)
    missing_count = sum(1 for v in missing.values() if v)
    confidence = max(0.55, 0.95 - uncertainty_score * 0.25) if missing_count else 0.95
    reason = (
        f"{missing_count} critical fields missing with contextual sparse-profile risk"
        if score else "Critical fields are sufficiently present"
    )
    return _rr("contextual_missingness", score, confidence, reason,
               {"missing_critical_fields": [k for k, v in missing.items() if v],
                "profession_cluster": cluster, "core_sparse": core_sparse,
                "activity_sparse": activity_sparse,
                "uncertainty_only_score": uncertainty_score,
                "suspicious_missingness_score": clamp01(suspicious_score),
                "neutral_missing_fields": neutral_missing},
               uncertainty_score,
               raw_signal_value=missing_count, normalized_signal_value=score,
               threshold_used=0.0,
               missing_fields=[k for k, v in missing.items() if v],
               abstained_reason=None if score > 0 else "threshold_not_met",
               missingness_penalty=uncertainty_score)


def rule_interaction_salary_experience(features: dict) -> RuleResult:
    c = features.get("career", {})
    income = c.get("annual_income_lpa")
    years_exp = c.get("years_experience")
    cluster = c.get("profession_cluster") or "other"
    missing = _missing_fields({"annual_income_lpa": income, "years_experience": years_exp})
    conf = _available_confidence(income, years_exp)
    if missing:
        return _rr("interaction_salary_experience", 0.0, conf,
                   "Income or experience missing for interaction",
                   {"income": income, "years_experience": years_exp}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    soft_min, soft_max = expected_salary_range_by_experience(years_exp)
    overshoot = (income - soft_max) / max(soft_max, 1.0)
    early_career = years_exp <= 3
    score = _linear_ramp(overshoot, 0.15, 1.2, ceiling=0.95) if early_career and overshoot > 0 else 0.0
    if cluster in {"business_entrepreneur"}:
        score *= 0.55
        conf *= 0.75
    return _rr("interaction_salary_experience", score, conf,
               "High salary compounds with low experience" if score else "Salary-experience interaction not active",
               {"income": income, "years_experience": years_exp,
                "expected_max": soft_max, "early_career": early_career},
               raw_signal_value=overshoot, normalized_signal_value=score,
               threshold_used=0.15)


def rule_interaction_education_income_trajectory(features: dict) -> RuleResult:
    c = features.get("career", {})
    d = features.get("derived", {})
    edu = c.get("education_level_normalized")
    edu_rank = c.get("education_rank")
    income = c.get("annual_income_lpa")
    income_per_exp = d.get("income_per_experience")
    missing = _missing_fields({"education_rank": edu_rank, "annual_income_lpa": income})
    conf = _available_confidence(edu_rank, income)
    if missing:
        return _rr("interaction_education_income_trajectory", 0.0, conf,
                   "Education or income missing for trajectory interaction",
                   {"education_level": edu, "income": income}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    high_income_low_edu = edu_rank <= 2.5 and income >= 30
    steep_trajectory = income_per_exp is not None and income_per_exp >= 18 and edu_rank <= 3.0
    raw = max(
        (income - 30) / 50 if high_income_low_edu else 0.0,
        (income_per_exp - 18) / 40 if steep_trajectory else 0.0,
    )
    score = _linear_ramp(raw, 0.0, 1.0, ceiling=0.88)
    return _rr("interaction_education_income_trajectory", score, conf,
               "Education-income trajectory is unusually steep" if score else "Education-income trajectory not unusual",
               {"education_level": edu, "education_rank": edu_rank,
                "income": income, "income_per_experience": income_per_exp},
               raw_signal_value=raw, normalized_signal_value=score,
               threshold_used="low_edu_high_income_or_steep_income_per_exp")


def rule_interaction_text_email(features: dict) -> RuleResult:
    e = features.get("email", {})
    t = features.get("textual", {})
    domain = e.get("email_domain")
    word_count = t.get("bio_word_count")
    disposable = bool(e.get("disposable_domain") or e.get("typo_domain"))
    sparse_text = word_count is not None and word_count < 12
    generic_hits = t.get("generic_language_hits") or 0
    conf = _available_confidence(domain, word_count)
    missing = _missing_fields({"email_domain": domain, "bio_word_count": word_count})
    if missing:
        return _rr("interaction_text_email", 0.0, conf,
                   "Email or text missing for text-email interaction",
                   {"email_domain": domain, "bio_word_count": word_count}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    raw = int(disposable) + int(sparse_text) + (0.5 if generic_hits >= 3 else 0.0)
    score = 0.0
    if disposable and (sparse_text or generic_hits >= 3):
        score = min(0.82, 0.42 + 0.16 * raw)
    return _rr("interaction_text_email", score, conf,
               "Sparse/generic text compounds with disposable or typo email" if score else "Text-email interaction not active",
               {"email_domain": domain, "bio_word_count": word_count,
                "generic_language_hits": generic_hits, "domain_low_quality": disposable},
               raw_signal_value=raw, normalized_signal_value=score,
               threshold_used="domain_low_quality_and_sparse_text")


def rule_interaction_income_behavior(features: dict) -> RuleResult:
    c = features.get("career", {})
    b = features.get("behavioral", {})
    income = c.get("annual_income_lpa")
    account_age = b.get("account_age_days")
    edit_count = b.get("profile_edit_count") or 0
    total_logins = b.get("n_total_logins") or 0
    missing = _missing_fields({"annual_income_lpa": income, "account_age_days": account_age})
    conf = _available_confidence(income, account_age)
    if missing:
        return _rr("interaction_income_behavior", 0.0, conf,
                   "Income or account age missing for income-behavior interaction",
                   {"income": income, "account_age_days": account_age}, 1.0 - conf,
                   missing_fields=missing, missingness_penalty=1.0 - conf)
    high_income = income >= 45
    unstable = (account_age <= 14 and edit_count >= 3) or (account_age <= 30 and total_logins <= 2)
    raw = (income - 45) / 80 if high_income and unstable else 0.0
    score = _linear_ramp(raw, 0.0, 1.0, ceiling=0.72)
    return _rr("interaction_income_behavior", score, conf,
               "High claimed income has weak or unstable behavioral support" if score else "Income-behavior interaction not active",
               {"income": income, "account_age_days": account_age,
                "profile_edit_count": edit_count, "n_total_logins": total_logins},
               raw_signal_value=raw, normalized_signal_value=score,
               threshold_used="high_income_unstable_activity")


def rule_interaction_sparse_inconsistency(features: dict) -> RuleResult:
    d = features.get("derived", {})
    m = features.get("missingness", {})
    b = features.get("behavioral", {})
    n_flags = d.get("n_consistency_flags") or 0
    missing_critical = m.get("missing_critical_fields_count") or 0
    sparse_activity = (b.get("n_total_logins") in (None, 0, 1)) and (b.get("profile_edit_count") in (None, 0))
    raw = n_flags + 0.35 * missing_critical + (1.0 if sparse_activity else 0.0)
    # Phase 1: Lowered threshold from 1.0 to 0.5 to activate this high-quality rule more often
    # This rule has 3.9x lift when it fires - we want to capture more instances
    score = _linear_ramp(raw, 0.5, 3.0, ceiling=0.84)
    conf = 0.85
    return _rr("interaction_sparse_inconsistency", score, conf,
               "Sparse profile compounds with consistency problems" if score else "Sparse-inconsistency interaction not active",
               {"n_consistency_flags": n_flags,
                "missing_critical_fields_count": missing_critical,
                "sparse_activity": sparse_activity},
               raw_signal_value=raw, normalized_signal_value=score,
               threshold_used=0.5)


RULE_REGISTRY = (
    RuleSpec("age_experience", "Age and experience plausibility",
             RULE_WEIGHTS["age_experience"], RULE_EVIDENCE_WEIGHTS["age_experience"],
             ("demographic", "career", "derived"), rule_age_experience),
    RuleSpec("age_education", "Age and education plausibility",
             RULE_WEIGHTS["age_education"], RULE_EVIDENCE_WEIGHTS["age_education"],
             ("demographic", "career"), rule_age_education),
    RuleSpec("education_profession", "Education and profession compatibility",
             RULE_WEIGHTS["education_profession"], RULE_EVIDENCE_WEIGHTS["education_profession"],
             ("career",), rule_education_profession),
    RuleSpec("profession_email", "Profession and email consistency",
             RULE_WEIGHTS["profession_email"], RULE_EVIDENCE_WEIGHTS["profession_email"],
             ("career", "email"), rule_profession_email),
    RuleSpec("salary_experience", "Salary and experience plausibility",
             RULE_WEIGHTS["salary_experience"], RULE_EVIDENCE_WEIGHTS["salary_experience"],
             ("career",), rule_salary_experience),
    RuleSpec("salary_profession", "Salary and profession plausibility",
             RULE_WEIGHTS["salary_profession"], RULE_EVIDENCE_WEIGHTS["salary_profession"],
             ("career", "derived"), rule_salary_profession),
    RuleSpec("education_salary", "Education and salary plausibility",
             RULE_WEIGHTS["education_salary"], RULE_EVIDENCE_WEIGHTS["education_salary"],
             ("career",), rule_education_salary),
    RuleSpec("company_income", "Company tier and income alignment",
             RULE_WEIGHTS["company_income"], RULE_EVIDENCE_WEIGHTS["company_income"],
             ("career", "derived"), rule_company_income),
    RuleSpec("email_quality", "Email quality risk",
             RULE_WEIGHTS["email_quality"], RULE_EVIDENCE_WEIGHTS["email_quality"],
             ("email",), rule_email_quality),
    RuleSpec("behavioral_integrity", "Behavioral consistency risk",
             RULE_WEIGHTS["behavioral_integrity"], RULE_EVIDENCE_WEIGHTS["behavioral_integrity"],
             ("behavioral",), rule_behavioral_integrity),
    RuleSpec("textual_quality", "Sparse or generic text risk",
             RULE_WEIGHTS["textual_quality"], RULE_EVIDENCE_WEIGHTS["textual_quality"],
             ("textual",), rule_textual_quality),
    RuleSpec("contextual_missingness", "Context-aware missing critical evidence",
             RULE_WEIGHTS["contextual_missingness"], RULE_EVIDENCE_WEIGHTS["contextual_missingness"],
             ("career", "behavioral", "missingness"), rule_contextual_missingness),
    RuleSpec("interaction_salary_experience", "Interaction: high salary and low experience",
             RULE_WEIGHTS["interaction_salary_experience"], RULE_EVIDENCE_WEIGHTS["interaction_salary_experience"],
             ("career",), rule_interaction_salary_experience),
    RuleSpec("interaction_education_income_trajectory", "Interaction: education and income trajectory",
             RULE_WEIGHTS["interaction_education_income_trajectory"], RULE_EVIDENCE_WEIGHTS["interaction_education_income_trajectory"],
             ("career", "derived"), rule_interaction_education_income_trajectory),
    RuleSpec("interaction_text_email", "Interaction: sparse text and weak email",
             RULE_WEIGHTS["interaction_text_email"], RULE_EVIDENCE_WEIGHTS["interaction_text_email"],
             ("email", "textual"), rule_interaction_text_email),
    RuleSpec("interaction_income_behavior", "Interaction: high income and weak behavioral metadata",
             RULE_WEIGHTS["interaction_income_behavior"], RULE_EVIDENCE_WEIGHTS["interaction_income_behavior"],
             ("career", "behavioral"), rule_interaction_income_behavior),
    RuleSpec("interaction_sparse_inconsistency", "Interaction: profile inconsistency and sparse history",
             RULE_WEIGHTS["interaction_sparse_inconsistency"], RULE_EVIDENCE_WEIGHTS["interaction_sparse_inconsistency"],
             ("derived", "missingness", "behavioral"), rule_interaction_sparse_inconsistency),
)


# Compatibility wrappers for older direct imports.
def score_age_experience(age: int, years_exp: int) -> float:
    return rule_age_experience({
        "demographic": {"age": age},
        "career": {"years_experience": years_exp, "profession_cluster": "other"},
        "derived": {"career_start_age": age - years_exp if age is not None and years_exp is not None else None},
    }).score


def score_age_education(age: int, education_level: str) -> float:
    return rule_age_education({
        "demographic": {"age": age},
        "career": {"education_level_normalized": education_level},
    }).score


def score_education_profession(education_level: str, profession: str) -> float:
    return rule_education_profession({
        "career": {
            "education_level_normalized": education_level,
            "education_rank": {"10th": 1, "12th": 2, "Graduate": 3, "Post Graduate": 4, "PhD": 5}.get(education_level),
            "profession_normalized": profession,
            "profession_cluster": "other",
        }
    }).score


def score_profession_email(profession: str, email_domain: str) -> float:
    return rule_profession_email({
        "career": {"profession_normalized": profession, "profession_cluster": "other"},
        "email": {"email_domain": email_domain, "disposable_domain": False},
    }).score


def score_salary_experience(annual_income_lpa: float, years_exp: int) -> float:
    return rule_salary_experience({
        "career": {"annual_income_lpa": annual_income_lpa, "years_experience": years_exp, "profession_cluster": "other"}
    }).score


def score_salary_profession(annual_income_lpa: float, profession: str) -> float:
    return rule_salary_profession({
        "career": {"annual_income_lpa": annual_income_lpa, "profession_normalized": profession, "profession_cluster": "other"},
        "derived": {"expected_income_band": "within_expected"},
    }).score


def score_education_salary(education_level: str, annual_income_lpa: float) -> float:
    return rule_education_salary({
        "career": {"education_level_normalized": education_level, "annual_income_lpa": annual_income_lpa}
    }).score

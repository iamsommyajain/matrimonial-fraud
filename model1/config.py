"""
Central configuration for Model 1 functional consistency scoring.

All values are deterministic heuristics.  Tune here first, then inspect
evaluate_m1.py rule analytics before changing rule code.
"""

from __future__ import annotations


RISK_THRESHOLDS = {
    "low": (0.00, 0.30),
    "medium": (0.30, 0.55),
    "high": (0.55, 0.75),
    "critical": (0.75, 1.01),
}

PAIR_FLAG_THRESHOLD = 0.45  # Lowered from 0.55 to capture more signals from high-quality rules
FUSION_STRATEGY = "probabilistic_sum"
SATURATION_ALPHA = 3.2
CONTRIBUTION_TEMPERATURE = 1.0
CONFIDENCE_FLOOR = 0.35
MINIMUM_SIGNAL_ACTIVATION = 0.015
SCORE_CAP = 0.98

EDU_RANK = {
    "10th": 1.0,
    "12th": 2.0,
    "Diploma": 2.5,
    "Graduate": 3.0,
    "Professional Graduate": 3.5,
    "Post Graduate": 4.0,
    "PhD": 5.0,
}

EDU_MIN_COMPLETION_AGE = {
    "10th": 14,
    "12th": 16,
    "Diploma": 17,
    "Graduate": 20,
    "Professional Graduate": 23,
    "Post Graduate": 22,
    "PhD": 26,
}

EDU_INCOME_RANGE = {
    "10th": (0.0, 8.0),
    "12th": (0.5, 12.0),
    "Diploma": (1.0, 15.0),
    "Graduate": (2.5, 40.0),
    "Professional Graduate": (4.0, 90.0),
    "Post Graduate": (4.0, 90.0),
    "PhD": (4.0, 70.0),
}

CLUSTER_MIN_EDU_RANK = {
    "medicine": 3.5,
    "law": 3.0,
    "finance_banking": 3.0,
    "teaching_academia": 3.0,
    "government_defense": 3.0,
    "data_analytics": 3.0,
    "software_engineering": 2.5,
    "management_consulting": 3.0,
    "nursing_paramedic": 2.0,
}

CLUSTER_INCOME_RANGE = {
    "software_engineering": (3.0, 80.0, 15.0),
    "data_analytics": (4.0, 60.0, 14.0),
    "medicine": (4.0, 100.0, 15.0),
    "nursing_paramedic": (1.5, 12.0, 4.0),
    "finance_banking": (5.0, 120.0, 18.0),
    "finance_accounting": (2.5, 30.0, 7.0),
    "law": (2.0, 80.0, 10.0),
    "teaching_academia": (1.5, 25.0, 6.0),
    "government_defense": (4.0, 30.0, 10.0),
    "business_entrepreneur": (0.0, 500.0, 20.0),
    "management_consulting": (6.0, 100.0, 18.0),
    "sales_marketing": (2.0, 30.0, 8.0),
    "design_creative": (1.5, 25.0, 7.0),
    "hr_admin": (2.0, 20.0, 6.0),
    "logistics_operations": (2.0, 20.0, 6.0),
    "student": (0.0, 3.0, 0.0),
    "other": (0.0, 500.0, 10.0),
}

MIN_START_AGE = {
    "medicine": 23,
    "law": 22,
    "software_engineering": 20,
    "data_analytics": 20,
    "finance_banking": 21,
    "finance_accounting": 21,
    "teaching_academia": 22,
    "government_defense": 21,
    "nursing_paramedic": 20,
    "business_entrepreneur": 18,
    "management_consulting": 20,
    "sales_marketing": 18,
    "design_creative": 18,
    "hr_admin": 19,
    "logistics_operations": 18,
    "student": 16,
    "other": 18,
}

HIGH_CONCERN_CLUSTERS = frozenset({
    "government_defense",
    "medicine",
    "teaching_academia",
    "finance_banking",
})

PERSONAL_EMAIL_DOMAINS = frozenset({
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "rediffmail.com",
    "yahoo.in",
    "icloud.com",
    "protonmail.com",
})

CRITICAL_FIELD_PRIORS = {
    "age": 0.10,
    "education_level": 0.16,
    "profession": 0.16,
    "annual_income_lpa": 0.13,
    "years_experience": 0.10,
    "email": 0.18,
}

FEATURE_CRITICALITY = {
    "critical": frozenset({"age", "education_level", "profession", "annual_income_lpa", "years_experience", "email"}),
    "supporting": frozenset({"company_tier", "college_tier", "login_timestamps", "profile_edit_count", "bio_text"}),
    "optional": frozenset({"hobbies", "partner_preferences", "about_family"}),
}

RULE_WEIGHTS = {
    "age_experience": 0.22,  # Increased from 0.18 (high-quality rule with good precision)
    "age_education": 0.10,
    "education_profession": 0.18,  # Increased from 0.15 (excellent separation: 20.5x odds ratio)
    "profession_email": 0.08,
    "salary_experience": 0.08,  # Reduced from 0.15 (too noisy: 27% firing rate on both fraud & legit)
    "salary_profession": 0.13,
    "education_salary": 0.09,
    "company_income": 0.05,
    "email_quality": 0.10,
    "behavioral_integrity": 0.07,
    "textual_quality": 0.04,
    "contextual_missingness": 0.16,
    "interaction_salary_experience": 0.22,  # Increased from 0.18 (3.9x lift when fires)
    "interaction_education_income_trajectory": 0.11,
    "interaction_text_email": 0.12,
    "interaction_income_behavior": 0.10,
    "interaction_sparse_inconsistency": 0.18,  # Increased from 0.14 (excellent quality)
}

RULE_EVIDENCE_WEIGHTS = {
    "age_experience": 0.92,
    "age_education": 0.82,
    "education_profession": 0.78,
    "profession_email": 0.70,
    "salary_experience": 0.84,
    "salary_profession": 0.75,
    "education_salary": 0.68,
    "company_income": 0.55,
    "email_quality": 0.78,
    "behavioral_integrity": 0.62,
    "textual_quality": 0.42,
    "contextual_missingness": 0.70,
    "interaction_salary_experience": 0.88,
    "interaction_education_income_trajectory": 0.74,
    "interaction_text_email": 0.76,
    "interaction_income_behavior": 0.70,
    "interaction_sparse_inconsistency": 0.78,
}


def expected_salary_range_by_experience(years_exp: int | float | None) -> tuple[float, float]:
    if years_exp is None:
        return (0.0, 500.0)
    if years_exp <= 1:
        return (2.0, 15.0)
    if years_exp <= 3:
        return (3.0, 20.0)
    if years_exp <= 5:
        return (6.0, 28.0)
    if years_exp <= 10:
        return (12.0, 45.0)
    return (18.0, 90.0)

"""
fuzzy_rules.py

One function per consistency pair.
All functions return a float in [0.0, 1.0]:
    0.0 = perfectly clean / expected
    1.0 = maximally suspicious / inconsistent

Design principles:
  - Smooth penalty curves (sigmoid / linear ramp), never hard binary
  - Two-sided where relevant (too high AND too low can both be suspicious)
  - Constants at top so tuning doesn't require reading the logic
"""

import math


# ─────────────────────────────────────────────────────────────────────────────
# Shared sigmoid helper
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid_penalty(x, center, steepness=1.0):
    """
    Returns a value in (0, 1) that rises as x moves away from center.
    Used to create smooth penalties around a threshold.
    steepness controls how sharp the curve is.
    """
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


def _linear_ramp(value, ok_below, max_penalty_at, floor=0.0, ceiling=1.0):
    """
    Returns 0.0 if value <= ok_below.
    Rises linearly to ceiling as value approaches max_penalty_at.
    Clamps at ceiling beyond that.
    """
    if value <= ok_below:
        return floor
    if value >= max_penalty_at:
        return ceiling
    ratio = (value - ok_below) / (max_penalty_at - ok_below)
    return floor + ratio * (ceiling - floor)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 1: Age ↔ Experience
# ─────────────────────────────────────────────────────────────────────────────
# Minimum working age after different education levels (conservative):
#   10th / 12th  → 16-18
#   Graduate     → 21
#   Post Grad    → 23
#   PhD          → 27
# We use the most lenient assumption: you could start work at 18.
# So max realistic experience = age - 18.

AGE_EXP_MIN_WORK_AGE = 18          # absolute floor
AGE_EXP_SOFT_BUFFER  = 1           # 1 year slack before penalty starts
AGE_EXP_STEEPNESS    = 0.6         # how sharp the sigmoid rises


def score_age_experience(age: int, years_exp: int) -> float:
    """
    Penalises profiles where years_exp is impossible or implausible given age.

    Impossible:  exp > age - 18          → hard upper ceiling
    Implausible: exp > age - 18 - buffer → soft ramp
    """
    if age <= 0 or years_exp < 0:
        return 0.5   # missing data → neutral

    max_possible = max(0, age - AGE_EXP_MIN_WORK_AGE)

    if years_exp <= 0:
        return 0.0   # 0 experience is always fine

    # How many years over the realistic max?
    overshoot = years_exp - max_possible

    if overshoot <= 0:
        return 0.0   # within range, clean

    # Soft ramp: 1 year over → small penalty, 5+ years over → near 1.0
    return _linear_ramp(overshoot, AGE_EXP_SOFT_BUFFER, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 2: Age ↔ Education
# ─────────────────────────────────────────────────────────────────────────────
# Minimum realistic ages to *complete* each education level:
#   10th        → 15
#   12th        → 17
#   Graduate    → 20
#   Post Grad   → 22
#   PhD         → 26

EDU_MIN_COMPLETION_AGE = {
    "10th":          15,
    "12th":          17,
    "Graduate":      20,
    "Post Graduate": 22,
    "PhD":           26,
}
AGE_EDU_STEEPNESS = 0.8


def score_age_education(age: int, education_level: str) -> float:
    """
    Penalises profiles where the claimed education is impossible for their age.
    Example: PhD at age 23 → suspicious. PhD at age 20 → near impossible.
    """
    min_age = EDU_MIN_COMPLETION_AGE.get(education_level)
    if min_age is None or age <= 0:
        return 0.0   # unknown education level → no penalty

    if age >= min_age:
        return 0.0   # old enough, clean

    # How many years short of the minimum?
    shortfall = min_age - age
    # 1 year short → small penalty, 5+ years short → near 1.0
    return _linear_ramp(shortfall, 0, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 3: Education ↔ Profession
# ─────────────────────────────────────────────────────────────────────────────
# Hard-requirement professions: you cannot do these without a minimum degree.
# Maps profession → minimum required education level (as an index 0-4).

EDU_ORDER = ["10th", "12th", "Graduate", "Post Graduate", "PhD"]

# (minimum_edu_index, penalty_if_below)
# penalty_if_below is the score returned if education is insufficient.
# We keep it < 1.0 because self-employment and exceptions exist.
PROFESSION_EDU_REQUIREMENTS = {
    # Profession                  min_edu_idx  penalty
    "Doctor":                     (3,          0.95),   # needs MD / MBBS (PG)
    "Lawyer":                     (2,          0.85),   # needs LLB (Graduate)
    "Chartered Accountant":       (2,          0.85),
    "Research Scientist":         (3,          0.85),
    "Professor":                  (3,          0.80),
    "Senior Software Engineer":   (2,          0.65),
    "Software Engineer":          (2,          0.55),
    "Data Scientist":             (2,          0.65),
    "Product Manager":            (2,          0.60),
    "Business Analyst":           (2,          0.55),
    "Finance Manager":            (2,          0.55),
    "Government IAS/IPS":         (2,          0.90),
    "Principal Engineer":         (3,          0.70),
}
EDU_PROF_DEFAULT_PENALTY = 0.40   # for professions not in the map


def score_education_profession(education_level: str, profession: str) -> float:
    """
    Penalises profiles where the education is insufficient for the profession.
    """
    if education_level not in EDU_ORDER:
        return 0.0

    edu_idx = EDU_ORDER.index(education_level)
    req = PROFESSION_EDU_REQUIREMENTS.get(profession)

    if req is None:
        # Profession not in our requirement map — low default penalty if
        # education is extremely low (10th claiming senior roles)
        if edu_idx == 0 and profession not in [
            "Shop Owner", "Daily Wage Worker", "Driver",
            "Security Guard", "Domestic Worker", "Delivery Executive"
        ]:
            return EDU_PROF_DEFAULT_PENALTY
        return 0.0

    min_edu_idx, penalty = req
    if edu_idx >= min_edu_idx:
        return 0.0   # meets requirement

    # Below requirement — how far below?
    shortfall = min_edu_idx - edu_idx
    # 1 level below → half the penalty; 2+ levels below → full penalty
    return penalty * min(1.0, shortfall / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 4: Profession ↔ Email Domain
# ─────────────────────────────────────────────────────────────────────────────
# Government / institutional roles should ideally use official domains.
# Gmail / Yahoo for such roles is a soft penalty (many real employees use
# personal email on matrimonial sites).

PERSONAL_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "rediffmail.com", "yahoo.in", "icloud.com",
}

GOV_INSTITUTIONAL_DOMAINS = {
    "gov.in", "nic.in", "iit.ac.in", "nit.ac.in", "aiims.edu",
    "army.mil", "ips.gov.in", "ias.gov.in",
}

SUSPICIOUS_DOMAINS = {
    "mailinator.com", "tempmail.com", "guerrillamail.com", "throwam.com",
    "fakeinbox.com", "yopmail.com", "trashmail.com", "dispostable.com",
}

# Professions where personal email on matrimonial site is totally fine
LOW_CONCERN_PROFESSIONS = {
    "Shop Owner", "Business Owner", "Sales Executive", "Teacher",
    "HR Executive", "Marketing Executive", "Accountant", "Nurse",
    "Pharmacist", "Bank Clerk", "Office Assistant",
}

# Professions where institutional email would be expected / verifiable
HIGH_CONCERN_PROFESSIONS = {
    "Government IAS/IPS", "Doctor", "Professor", "Research Scientist",
    "Government Employee", "Principal Engineer",
}

PROF_EMAIL_SOFT_PENALTY   = 0.20   # personal email for high-concern profession
PROF_EMAIL_SUSPICIOUS_PENALTY = 0.85  # throwaway domain


def score_profession_email(profession: str, email_domain: str) -> float:
    """
    Soft penalty for high-status/government professions using personal email.
    Hard penalty for throwaway/suspicious domains (anyone).
    """
    domain = email_domain.lower().strip()

    # Throwaway domain → always suspicious regardless of profession
    if domain in SUSPICIOUS_DOMAINS:
        return PROF_EMAIL_SUSPICIOUS_PENALTY

    # Personal email for high-concern profession → soft penalty
    if profession in HIGH_CONCERN_PROFESSIONS and domain in PERSONAL_EMAIL_DOMAINS:
        return PROF_EMAIL_SOFT_PENALTY

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Rule 5: Salary ↔ Experience  (two-sided)
# ─────────────────────────────────────────────────────────────────────────────
# Indian market 2024 rough benchmarks (LPA):
#   0-2 years exp  → 3-12 LPA typical
#   3-5 years exp  → 8-20 LPA typical
#   6-10 years exp → 15-35 LPA typical
#   10+ years exp  → 25-60 LPA typical (with wide variance for senior roles)

def _expected_salary_range(years_exp: int) -> tuple:
    """Returns (soft_min, soft_max) LPA for a given experience level."""
    if years_exp <= 1:
        return (2.0, 15.0)
    elif years_exp <= 3:
        return (3.0, 20.0)
    elif years_exp <= 5:
        return (6.0, 28.0)
    elif years_exp <= 10:
        return (12.0, 45.0)
    else:
        return (18.0, 80.0)


SALARY_EXP_HIGH_STEEPNESS = 0.15   # how sharp the upper penalty rises
SALARY_EXP_LOW_WEIGHT     = 0.40   # low salary is softer signal than high


def score_salary_experience(annual_income_lpa: float, years_exp: int) -> float:
    """
    Two-sided penalty:
      - Way too high for experience → strong fraud signal
      - Way too low for experience → soft signal (startup equity, NGO, etc.)
    """
    if annual_income_lpa <= 0 or years_exp < 0:
        return 0.0

    soft_min, soft_max = _expected_salary_range(years_exp)

    score = 0.0

    # Upper side: salary far above range
    if annual_income_lpa > soft_max:
        overshoot_ratio = (annual_income_lpa - soft_max) / soft_max
        upper_penalty = _linear_ramp(overshoot_ratio, 0.1, 1.5)
        score = max(score, upper_penalty)

    # Lower side: salary suspiciously below range (softer)
    if annual_income_lpa < soft_min:
        undershoot_ratio = (soft_min - annual_income_lpa) / soft_min
        lower_penalty = _linear_ramp(undershoot_ratio, 0.1, 1.0) * SALARY_EXP_LOW_WEIGHT
        score = max(score, lower_penalty)

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 6: Salary ↔ Profession  (two-sided)
# ─────────────────────────────────────────────────────────────────────────────
# Expected LPA ranges per profession (soft bounds — outside these raises flags)

PROFESSION_SALARY_RANGES = {
    "Software Engineer":        (4.0,  25.0),
    "Senior Software Engineer": (10.0, 50.0),
    "Product Manager":          (12.0, 60.0),
    "Data Scientist":           (8.0,  45.0),
    "Doctor":                   (6.0,  60.0),
    "Professor":                (5.0,  20.0),
    "Teacher":                  (2.0,  8.0),
    "Lawyer":                   (4.0,  50.0),
    "Chartered Accountant":     (5.0,  35.0),
    "Bank Officer":             (4.0,  18.0),
    "Government Employee":      (3.0,  15.0),
    "Government IAS/IPS":       (8.0,  20.0),
    "Business Owner":           (3.0,  80.0),   # wide range — legitimate variance
    "Shop Owner":               (1.5,  12.0),
    "Research Scientist":       (6.0,  30.0),
    "Business Analyst":         (5.0,  30.0),
    "Finance Manager":          (8.0,  40.0),
    "HR Executive":             (3.0,  15.0),
    "Sales Manager":            (4.0,  20.0),
    "Principal Engineer":       (20.0, 80.0),
}

SALARY_PROF_LOW_WEIGHT = 0.35   # low salary for role → softer signal


def score_salary_profession(annual_income_lpa: float, profession: str) -> float:
    """
    Two-sided: flags salary too high or too low for the claimed profession.
    Low-salary penalty is softer (startup equity, NGO, new venture).
    """
    if annual_income_lpa <= 0:
        return 0.0

    salary_range = PROFESSION_SALARY_RANGES.get(profession)
    if salary_range is None:
        return 0.0   # unknown profession → no penalty

    soft_min, soft_max = salary_range
    score = 0.0

    # Upper side
    if annual_income_lpa > soft_max:
        overshoot_ratio = (annual_income_lpa - soft_max) / soft_max
        score = max(score, _linear_ramp(overshoot_ratio, 0.1, 1.5))

    # Lower side (soft)
    if annual_income_lpa < soft_min:
        undershoot_ratio = (soft_min - annual_income_lpa) / soft_min
        score = max(score, _linear_ramp(undershoot_ratio, 0.1, 1.0) * SALARY_PROF_LOW_WEIGHT)

    return min(1.0, score)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 7: Education ↔ Salary
# ─────────────────────────────────────────────────────────────────────────────
# A PhD earning 2 LPA is suspicious the OTHER way — possible but unusual.
# A 10th-pass claiming 80 LPA is a strong fraud signal.

EDU_SALARY_RANGES = {
    "10th":          (1.0, 8.0),
    "12th":          (1.5, 12.0),
    "Graduate":      (3.0, 35.0),
    "Post Graduate": (5.0, 60.0),
    "PhD":           (6.0, 40.0),    # PhD in India often in academia — cap lower
}

EDU_SALARY_LOW_WEIGHT = 0.25   # over-qualified but low salary → very soft signal


def score_education_salary(education_level: str, annual_income_lpa: float) -> float:
    """
    Two-sided: high salary for low education is a strong flag.
    Low salary for high education is a soft flag.
    """
    if annual_income_lpa <= 0:
        return 0.0

    salary_range = EDU_SALARY_RANGES.get(education_level)
    if salary_range is None:
        return 0.0

    soft_min, soft_max = salary_range
    score = 0.0

    # Upper side — 10th pass claiming 80 LPA is very suspicious
    if annual_income_lpa > soft_max:
        overshoot_ratio = (annual_income_lpa - soft_max) / soft_max
        score = max(score, _linear_ramp(overshoot_ratio, 0.05, 1.2))

    # Lower side — very soft
    if annual_income_lpa < soft_min:
        undershoot_ratio = (soft_min - annual_income_lpa) / soft_min
        score = max(score, _linear_ramp(undershoot_ratio, 0.1, 1.0) * EDU_SALARY_LOW_WEIGHT)

    return min(1.0, score)

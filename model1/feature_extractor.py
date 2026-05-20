"""
feature_extractor.py  (v2.0)

Enhanced M1 feature extractor for the Matrimonial Fraud Detection System.

Architecture: flat extraction → grouped sub-extractors → structured output dict.
No ML models, no external APIs, no heavyweight NLP.

Output structure:
    {
        "demographic":  {...},   # age, height, location
        "career":       {...},   # education, profession, income, experience
        "email":        {...},   # domain risk, consistency, alignment
        "behavioral":   {...},   # login/edit/ip/activity patterns
        "media":        {...},   # photo/exif/deepfake signals
        "textual":      {...},   # bio/family/hobby heuristics
        "derived":      {...},   # consistency cross-signals
        "missingness":  {...},   # completeness score, missing indicators
        "_flat":        {...},   # backward-compat aliases for existing scorer
    }

Downstream scorer (scorer.py) should migrate to keyed sub-dicts over time.
The _flat block preserves the original 6-key contract in the interim.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime, timezone
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS & LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────────────

# Education: canonical label → ordinal rank
_EDU_RANK: dict[str, float] = {
    "10th":                  1.0,
    "12th":                  2.0,
    "Diploma":               2.5,
    "Graduate":              3.0,
    "Professional Graduate": 3.5,   # MBBS, LLB, B.Arch, BDS …
    "Post Graduate":         4.0,
    "PhD":                   5.0,
}

# Raw strings → canonical education label
_EDU_ALIASES: dict[str, str] = {
    # Secondary
    "10":               "10th",
    "10th":             "10th",
    "matriculation":    "10th",
    "sslc":             "10th",
    "x":                "10th",
    # Higher secondary
    "12":               "12th",
    "12th":             "12th",
    "intermediate":     "12th",
    "hsc":              "12th",
    "puc":              "12th",
    "xii":              "12th",
    # Diploma
    "diploma":          "Diploma",
    "polytechnic":      "Diploma",
    "iti":              "Diploma",
    # Graduate
    "graduate":         "Graduate",
    "graduation":       "Graduate",
    "ug":               "Graduate",
    "bachelor":         "Graduate",
    "bachelors":        "Graduate",
    "b.tech":           "Graduate",
    "btech":            "Graduate",
    "b.e":              "Graduate",
    "be":               "Graduate",
    "b.sc":             "Graduate",
    "bsc":              "Graduate",
    "b.com":            "Graduate",
    "bcom":             "Graduate",
    "b.a":              "Graduate",
    "ba":               "Graduate",
    "b.b.a":            "Graduate",
    "bba":              "Graduate",
    "b.ca":             "Graduate",
    "bca":              "Graduate",
    # Professional Graduate (5+ year clinical/professional degrees)
    "mbbs":             "Professional Graduate",
    "bds":              "Professional Graduate",
    "b.arch":           "Professional Graduate",
    "llb":              "Professional Graduate",
    "b.pharm":          "Professional Graduate",
    "bpharm":           "Professional Graduate",
    # Post Graduate
    "post graduate":    "Post Graduate",
    "postgraduate":     "Post Graduate",
    "pg":               "Post Graduate",
    "masters":          "Post Graduate",
    "master":           "Post Graduate",
    "mba":              "Post Graduate",
    "m.tech":           "Post Graduate",
    "mtech":            "Post Graduate",
    "m.sc":             "Post Graduate",
    "msc":              "Post Graduate",
    "m.com":            "Post Graduate",
    "mcom":             "Post Graduate",
    "m.a":              "Post Graduate",
    "ma":               "Post Graduate",
    "mca":              "Post Graduate",
    "m.b.a":            "Post Graduate",
    "ll.m":             "Post Graduate",
    "llm":              "Post Graduate",
    "m.d":              "Post Graduate",   # Medical PG
    "ms":               "Post Graduate",   # MS Surgery
    # PhD
    "phd":              "PhD",
    "ph.d":             "PhD",
    "ph.d.":            "PhD",
    "doctorate":        "PhD",
    "doctoral":         "PhD",
    "d.sc":             "PhD",
}

# Profession → cluster
_PROFESSION_CLUSTERS: dict[str, list[str]] = {
    "software_engineering": [
        "software engineer", "software developer", "swe", "backend developer",
        "frontend developer", "full stack developer", "full stack engineer",
        "fullstack developer", "web developer", "mobile developer",
        "android developer", "ios developer", "devops engineer", "sre",
        "data engineer", "ml engineer", "machine learning engineer",
        "ai engineer", "platform engineer", "infrastructure engineer",
        "qa engineer", "test engineer", "embedded engineer", "game developer",
        "software architect", "tech lead", "engineering manager",
    ],
    "data_analytics": [
        "data scientist", "data analyst", "business analyst",
        "quantitative analyst", "research analyst", "bi developer",
        "analytics engineer",
    ],
    "medicine": [
        "doctor", "physician", "surgeon", "mbbs", "dentist", "bds",
        "radiologist", "cardiologist", "dermatologist", "pediatrician",
        "gynaecologist", "gynecologist", "neurologist", "oncologist",
        "psychiatrist", "anesthesiologist", "pathologist", "ophthalmologist",
        "orthopedic", "ent specialist", "general practitioner", "gp",
        "medical officer", "resident doctor", "intern doctor",
    ],
    "nursing_paramedic": [
        "nurse", "nursing officer", "staff nurse", "paramedic",
        "physiotherapist", "occupational therapist", "dietitian",
        "pharmacist",
    ],
    "finance_banking": [
        "ca", "chartered accountant", "cpa", "banker", "investment banker",
        "financial analyst", "equity analyst", "portfolio manager",
        "risk analyst", "credit analyst", "actuary", "relationship manager",
        "branch manager", "loan officer", "treasury analyst",
        "fund manager", "hedge fund analyst",
    ],
    "finance_accounting": [
        "accountant", "accounts executive", "bookkeeper", "tax consultant",
        "auditor", "internal auditor", "cost accountant", "cma",
        "finance executive", "finance manager",
    ],
    "law": [
        "lawyer", "advocate", "attorney", "legal advisor", "solicitor",
        "barrister", "judge", "legal officer", "corporate lawyer",
        "criminal lawyer", "civil lawyer",
    ],
    "teaching_academia": [
        "teacher", "professor", "lecturer", "assistant professor",
        "associate professor", "principal", "headmaster", "tutor",
        "research scholar", "research associate", "postdoctoral researcher",
        "scientist", "researcher",
    ],
    "government_defense": [
        "ias", "ips", "ifs", "civil servant", "government officer",
        "bureaucrat", "army officer", "navy officer", "air force officer",
        "police officer", "defence officer", "psu employee",
        "bank employee", "central govt", "state govt",
    ],
    "business_entrepreneur": [
        "business owner", "entrepreneur", "founder", "co-founder",
        "director", "managing director", "ceo", "cto", "coo", "cfo",
        "business development", "bd manager",
    ],
    "management_consulting": [
        "consultant", "management consultant", "strategy consultant",
        "project manager", "product manager", "program manager",
        "scrum master", "operations manager", "supply chain manager",
    ],
    "sales_marketing": [
        "sales executive", "sales manager", "marketing executive",
        "marketing manager", "brand manager", "digital marketer",
        "content writer", "copywriter", "seo specialist",
        "social media manager",
    ],
    "design_creative": [
        "designer", "ui designer", "ux designer", "graphic designer",
        "product designer", "interior designer", "architect",
        "fashion designer", "animator", "video editor", "photographer",
    ],
    "hr_admin": [
        "hr", "human resources", "recruiter", "talent acquisition",
        "hr manager", "admin executive", "administrative officer",
    ],
    "logistics_operations": [
        "logistics", "supply chain", "warehouse manager",
        "operations executive", "fleet manager",
    ],
    "student": [
        "student", "college student", "mba student", "phd student",
        "research student",
    ],
    "other": [],   # fallback
}

# Build reverse map: canonical_lower → cluster
_PROFESSION_LOOKUP: dict[str, str] = {}
for _cluster, _terms in _PROFESSION_CLUSTERS.items():
    for _t in _terms:
        _PROFESSION_LOOKUP[_t.lower()] = _cluster

# Expected income ranges (LPA) per profession cluster
# (min_reasonable, max_reasonable, median_approx)
_CLUSTER_INCOME_RANGE: dict[str, tuple[float, float, float]] = {
    "software_engineering":  (3.0,  80.0,  15.0),
    "data_analytics":        (4.0,  60.0,  14.0),
    "medicine":              (4.0, 100.0,  15.0),
    "nursing_paramedic":     (1.5,  12.0,   4.0),
    "finance_banking":       (5.0, 120.0,  18.0),
    "finance_accounting":    (2.5,  30.0,   7.0),
    "law":                   (2.0,  80.0,  10.0),
    "teaching_academia":     (1.5,  25.0,   6.0),
    "government_defense":    (4.0,  30.0,  10.0),
    "business_entrepreneur": (0.0, 500.0,  20.0),
    "management_consulting": (6.0, 100.0,  18.0),
    "sales_marketing":       (2.0,  30.0,   8.0),
    "design_creative":       (1.5,  25.0,   7.0),
    "hr_admin":              (2.0,  20.0,   6.0),
    "logistics_operations":  (2.0,  20.0,   6.0),
    "student":               (0.0,   3.0,   0.0),
    "other":                 (0.0, 500.0,  10.0),
}

# Expected income ranges per education level (rough heuristic)
_EDU_INCOME_RANGE: dict[str, tuple[float, float]] = {
    "10th":                  (0.0,  4.0),
    "12th":                  (0.5,  6.0),
    "Diploma":               (1.0,  8.0),
    "Graduate":              (2.5, 40.0),
    "Professional Graduate": (4.0, 80.0),
    "Post Graduate":         (4.0, 80.0),
    "PhD":                   (5.0, 60.0),
}

# Minimum working age by profession cluster
_MIN_START_AGE: dict[str, int] = {
    "medicine":              23,   # MBBS + internship
    "law":                   22,
    "software_engineering":  20,
    "data_analytics":        20,
    "finance_banking":       21,
    "finance_accounting":    21,
    "teaching_academia":     22,
    "government_defense":    21,
    "nursing_paramedic":     20,
    "business_entrepreneur": 18,
    "management_consulting": 20,
    "sales_marketing":       18,
    "design_creative":       18,
    "hr_admin":              19,
    "logistics_operations":  18,
    "student":               16,
    "other":                 18,
}

# Disposable/temp email domains (sample — extend as needed)
_DISPOSABLE_DOMAINS: frozenset[str] = frozenset({
    "mailinator.com", "guerrillamail.com", "tempmail.com", "throwam.com",
    "trashmail.com", "yopmail.com", "sharklasers.com", "guerrillamailblock.com",
    "grr.la", "spam4.me", "maildrop.cc", "dispostable.com", "fakeinbox.com",
    "getairmail.com", "mailnull.com", "spamgourmet.com", "10minutemail.com",
    "discard.email", "mohmal.com", "crazymailing.com",
})

# Common typo-squatted domains
_TYPO_DOMAINS: frozenset[str] = frozenset({
    "gmial.com", "gmai.com", "gmaill.com", "gamil.com", "gmail.co",
    "yaho.com", "yahooo.com", "hotamil.com", "hotmali.com", "outook.com",
    "outlok.com", "redifmail.com", "rediffmial.com",
})

# Suspicious username keyword fragments
_SUSPICIOUS_USERNAME_WORDS: frozenset[str] = frozenset({
    "sexy", "hot", "rich", "nri", "vip", "elite", "premium", "lucky",
    "dollar", "money", "invest", "profit", "win", "prize", "gift",
    "angel", "sweetheart", "love", "marry", "bride", "groom",
})

_GENERIC_BIO_PHRASES: list[str] = [
    "looking for a life partner",
    "simple and honest",
    "god fearing",
    "down to earth",
    "family oriented",
    "simple girl",
    "simple boy",
    "caring and loving",
    "i am a simple",
    "i am looking",
    "working professional",
    "decent family",
    "very simple",
    "good looking",
]

_CRITICAL_FIELDS: tuple[str, ...] = (
    "age", "education_level", "profession", "annual_income_lpa",
    "years_experience", "email",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TOP-LEVEL EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_m1_features(profile: dict) -> dict:
    """
    Extract, normalise, and derive all M1-relevant features from a raw profile.

    The returned dict is structured into thematic sub-dicts plus a ``_flat``
    block that preserves the original 6-key contract for backward compatibility
    with the existing scorer.

    Args:
        profile: Raw profile dict as loaded from the dataset.

    Returns:
        Structured feature dict (see module docstring for full schema).
    """
    # ── raw helpers ──────────────────────────────────────────────────────────
    g = _getter(profile)   # safe field accessor

    # ── sub-extractions ──────────────────────────────────────────────────────
    demographic = _extract_demographic(g)
    career      = _extract_career(g)
    email       = _extract_email(g)
    behavioral  = _extract_behavioral(g)
    textual     = _extract_textual(g)
    missingness = _extract_missingness(profile)

    # ── cross-signal derivation (needs multiple sub-dicts) ───────────────────
    derived = _derive_consistency_signals(demographic, career, email)

    # ── backward-compat flat aliases ─────────────────────────────────────────
    flat = {
        "age":               demographic["age"],
        "education_level":   career["education_level_normalized"],
        "years_experience":  career["years_experience"],
        "profession":        career["profession_normalized"],
        "annual_income_lpa": career["annual_income_lpa"],
        "email_domain":      email["email_domain"],
    }

    return {
        "demographic":  demographic,
        "career":       career,
        "email":        email,
        "behavioral":   behavioral,
        "textual":      textual,
        "derived":      derived,
        "missingness":  missingness,
        "_flat":        flat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SUB-EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_demographic(g) -> dict:
    """Age, height, location, marital status, migration flag."""
    age    = _safe_cast(g("age"), int)
    height = _safe_cast(g("height_cm"), float)

    return {
        "age":            _clamp(age, 0, 120),
        "gender":         _normalise_str(g("gender")),
        "religion":       _normalise_str(g("religion")),
        "caste":          _normalise_str(g("caste")),
        "mother_tongue":  _normalise_str(g("mother_tongue")),
        "height_cm":      _clamp(height, 50.0, 250.0),
        "city":           _normalise_str(g("city")),
        "state":          _normalise_str(g("state")),
        "native_state":   _normalise_str(g("native_state")),
        "country":        _normalise_str(g("country")),
        "is_migrant":     _safe_bool(g("is_migrant")),
        "marital_status": _normalise_str(g("marital_status")),
    }


def _extract_career(g) -> dict:
    """Education, profession, income, experience — with derived ranks."""
    raw_edu        = _normalise_str(g("education_level")) or ""
    edu_normalised = _normalise_education(raw_edu)
    edu_rank       = _EDU_RANK.get(edu_normalised)

    raw_profession   = _normalise_str(g("profession")) or ""
    prof_normalised  = _normalise_profession(raw_profession)
    prof_cluster     = _cluster_profession(prof_normalised)

    years_exp   = _clamp(_safe_cast(g("years_experience"), int), 0, 60)
    income      = _clamp(_safe_cast(g("annual_income_lpa"), float), 0.0, None)

    raw_company_tier = _normalise_str(g("company_tier")) or ""
    company_tier     = _normalise_company_tier(raw_company_tier)

    return {
        # ── raw/normalised ────────────────────────────────────────────────
        "education_level_raw":        raw_edu,
        "education_level_normalized": edu_normalised,
        "education_rank":             edu_rank,         # float | None
        "education_field":            _normalise_str(g("education_field")),
        "college_name":               _normalise_str(g("college_name")),
        "college_tier":               _normalise_str(g("college_tier")),

        "profession_raw":             raw_profession,
        "profession_normalized":      prof_normalised,
        "profession_cluster":         prof_cluster,

        "company_name":               _normalise_str(g("company_name")),
        "company_tier":               company_tier,
        "company_tier_numeric":       _company_tier_to_int(company_tier),

        "years_experience":           years_exp,
        "annual_income_lpa":          income,
    }


def _extract_email(g) -> dict:
    """
    Email domain, disposability, typos, username signals, and risk fields
    from the pre-computed ``email_risk_features`` JSON column.
    """
    raw_email  = _safe_cast(g("email"), str) or ""
    raw_domain = _safe_cast(g("email_domain"), str) or ""

    # derive domain from email address if not directly given
    if not raw_domain and raw_email:
        raw_domain = _extract_domain(raw_email)
    domain = (raw_domain or "").lower().strip()

    username  = raw_email.split("@")[0].lower() if "@" in raw_email else raw_email.lower()

    # parse pre-computed risk features (JSON string or dict)
    risk = _parse_json_field(g("email_risk_features"), default={})
    consistency_score = _safe_cast(g("email_consistency_score"), float)

    return {
        "email_domain":                   domain or None,
        "email_raw":                      raw_email or None,

        # ── domain risk ───────────────────────────────────────────────────
        "disposable_domain":              domain in _DISPOSABLE_DOMAINS,
        "typo_domain":                    domain in _TYPO_DOMAINS,
        "is_free_email":                  _is_free_email(domain),

        # ── username heuristics ───────────────────────────────────────────
        "username_entropy":               _string_entropy(username),
        "excessive_digits":               _count_digits(username) >= 4,
        "digit_ratio":                    _digit_ratio(username),
        "suspicious_keywords_present":    _has_suspicious_keywords(username),

        # ── pre-computed risk fields (pass-through with safe defaults) ────
        "institutional_alignment_score":  _safe_float(risk.get("institutional_alignment_score")),
        "name_alignment_score":           _safe_float(risk.get("name_alignment_score")),
        "domain_matches_company":         _safe_bool(risk.get("domain_matches_company")),

        "email_consistency_score":        consistency_score,
    }


def _extract_behavioral(g) -> dict:
    """
    Login frequency, edit frequency, IP diversity, device consistency,
    photo upload burstiness, account age, and activity recency.
    All derived from lightweight heuristics on timestamp/IP list fields.
    """
    now = datetime.now(tz=timezone.utc)

    created_at    = _parse_datetime(g("created_at"))
    last_active   = _parse_datetime(g("last_active_at"))
    login_ts      = _parse_timestamp_list(g("login_timestamps"))
    edit_ts       = _parse_timestamp_list(g("edit_timestamps"))
    photo_dates   = _parse_timestamp_list(g("photo_upload_dates"))
    ip_list       = _parse_string_list(g("login_ip_list"))
    edit_count    = _safe_cast(g("profile_edit_count"), int)
    device_type   = _normalise_str(g("device_type"))

    account_age_days    = _days_between(created_at, now)
    activity_recency    = _days_between(last_active, now)

    # frequency per day (avoid division by zero)
    login_freq = (len(login_ts) / account_age_days) if account_age_days else None
    edit_freq  = (edit_count   / account_age_days)  if (account_age_days and edit_count is not None) else None

    # IP diversity: unique IPs / total logins
    n_unique_ips  = len(set(ip_list)) if ip_list else None
    n_total_logins = len(login_ts) if login_ts else None
    ip_diversity  = (n_unique_ips / n_total_logins) if (n_unique_ips and n_total_logins) else None

    # photo upload burstiness: std of inter-upload gaps (low std = bursty)
    photo_burstiness = _timestamp_burstiness(photo_dates)

    # edit burstiness
    edit_burstiness  = _timestamp_burstiness(edit_ts)

    return {
        "account_age_days":        account_age_days,
        "activity_recency_days":   activity_recency,
        "login_frequency":         login_freq,
        "edit_frequency":          edit_freq,
        "ip_diversity":            ip_diversity,
        "n_unique_ips":            n_unique_ips,
        "n_total_logins":          n_total_logins,
        "device_type":             device_type,
        "device_consistent":       True if device_type else None,   # placeholder; enrich if multi-device log exists
        "photo_upload_burstiness": photo_burstiness,
        "edit_burstiness":         edit_burstiness,
        "profile_edit_count":      edit_count,
    }





def _extract_textual(g) -> dict:
    """Lightweight NLP-free heuristics on free-text fields."""
    bio      = _safe_cast(g("bio_text"), str)       or ""
    family   = _safe_cast(g("about_family"), str)   or ""
    hobbies  = _safe_cast(g("hobbies"), str)        or ""
    partner  = _safe_cast(g("partner_preferences"), str) or ""

    bio_lower    = bio.lower()
    family_lower = family.lower()

    hobby_count = len([h.strip() for h in re.split(r"[,;|/]", hobbies) if h.strip()]) if hobbies else 0

    # repeated phrase check: find duplicate 4-gram sequences
    bio_words  = bio_lower.split()
    ngram_reps = _ngram_repetition_score(bio_words, n=4)

    # generic phrase density
    generic_hits = sum(1 for phrase in _GENERIC_BIO_PHRASES if phrase in bio_lower)
    generic_hits += sum(1 for phrase in _GENERIC_BIO_PHRASES if phrase in family_lower)

    return {
        "bio_length":                  len(bio),
        "family_section_length":       len(family),
        "partner_pref_length":         len(partner),
        "hobby_count":                 hobby_count,
        "bio_word_count":              len(bio_words),
        "bio_ngram_repetition_score":  ngram_reps,
        "generic_language_hits":       generic_hits,
        "excessive_generic_language":  generic_hits >= 3,
    }


def _extract_missingness(profile: dict) -> dict:
    """
    Profile completeness signals.
    Missing critical fields are explicit fraud-relevant features.
    """
    total_fields    = len(profile)
    missing_fields  = sum(1 for v in profile.values() if v is None or v == "")
    completeness    = 1.0 - (missing_fields / total_fields) if total_fields else 0.0

    missing_critical = {
        f"missing_{field}": (
            profile.get(field) is None or profile.get(field) == ""
        )
        for field in _CRITICAL_FIELDS
    }
    missing_critical_count = sum(1 for v in missing_critical.values() if v)

    return {
        "profile_completeness_score":   round(completeness, 4),
        "total_fields_observed":        total_fields,
        "missing_fields_count":         missing_fields,
        "missing_critical_fields_count": missing_critical_count,
        **missing_critical,
    }


def _derive_consistency_signals(demographic: dict, career: dict, email: dict) -> dict:
    """
    Cross-feature consistency signals for M1 fraud reasoning.
    Returns numeric / categorical signals — no scoring thresholds.
    """
    age           = demographic.get("age")
    years_exp     = career.get("years_experience")
    income        = career.get("annual_income_lpa")
    edu_rank      = career.get("education_rank")
    edu_level     = career.get("education_level_normalized")
    prof_cluster  = career.get("profession_cluster") or "other"
    company_tier  = career.get("company_tier_numeric")

    # ── 1. career_start_age ──────────────────────────────────────────────────
    career_start_age = (age - years_exp) if (age is not None and years_exp is not None) else None

    # ── 2. income_per_experience ─────────────────────────────────────────────
    income_per_exp = (income / years_exp) if (income is not None and years_exp and years_exp > 0) else None

    # ── 3. expected_income_band ──────────────────────────────────────────────
    min_inc, max_inc, med_inc = _CLUSTER_INCOME_RANGE.get(prof_cluster, (0.0, 999.0, 10.0))
    if income is not None:
        if income < min_inc:
            income_band = "below_expected"
        elif income > max_inc:
            income_band = "above_expected"
        else:
            income_band = "within_expected"
    else:
        income_band = None

    # income deviation ratio: how many sigma from median (rough)
    income_deviation_ratio = None
    if income is not None and med_inc > 0:
        income_deviation_ratio = (income - med_inc) / med_inc

    # ── 4. education_age_consistency ─────────────────────────────────────────
    # Minimum age someone could realistically finish a given education level
    _min_age_for_edu = {
        "10th": 14, "12th": 16, "Diploma": 17, "Graduate": 20,
        "Professional Graduate": 23, "Post Graduate": 22, "PhD": 26,
    }
    min_age_edu = _min_age_for_edu.get(edu_level)
    education_age_consistent = None
    if age is not None and min_age_edu is not None:
        education_age_consistent = age >= min_age_edu

    # ── 5. profession_income_alignment ───────────────────────────────────────
    profession_income_alignment = income_band   # reuse band signal

    # ── 6. education_income_alignment ────────────────────────────────────────
    edu_income_consistent = None
    if edu_level and income is not None:
        edu_min, edu_max = _EDU_INCOME_RANGE.get(edu_level, (0.0, 9999.0))
        edu_income_consistent = edu_min <= income <= edu_max

    # ── 7. experience_age_plausibility ───────────────────────────────────────
    min_start = _MIN_START_AGE.get(prof_cluster, 18)
    experience_age_plausible = None
    if career_start_age is not None:
        experience_age_plausible = career_start_age >= min_start

    # ── 8. company_tier_income_alignment ─────────────────────────────────────
    # Tier 1 companies → typically higher income; mismatch is suspicious
    company_income_consistent = None
    if company_tier is not None and income is not None:
        if company_tier == 1 and income < 5.0:
            company_income_consistent = False   # tier-1 but suspiciously low income
        elif company_tier == 3 and income > 50.0:
            company_income_consistent = False   # low-tier but suspiciously high income
        else:
            company_income_consistent = True

    # ── 9. Aggregate plausibility flag count (useful for scorer) ─────────────
    flags = [
        education_age_consistent is False,
        edu_income_consistent is False,
        experience_age_plausible is False,
        company_income_consistent is False,
        income_band == "above_expected",
    ]
    n_consistency_flags = sum(1 for f in flags if f)

    return {
        "career_start_age":              career_start_age,
        "income_per_experience":         income_per_exp,
        "expected_income_band":          income_band,
        "income_deviation_ratio":        income_deviation_ratio,
        "education_age_consistent":      education_age_consistent,
        "profession_income_alignment":   profession_income_alignment,
        "education_income_consistent":   edu_income_consistent,
        "experience_age_plausible":      experience_age_plausible,
        "company_income_consistent":     company_income_consistent,
        "n_consistency_flags":           n_consistency_flags,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_education(raw: str) -> str:
    """Map raw education string → canonical label via alias table."""
    key = raw.strip().lower()
    return _EDU_ALIASES.get(key, raw)   # unknown → pass through unchanged


def _normalise_profession(raw: str) -> str:
    """
    Clean and title-case profession string.
    Returns empty string for None/empty.
    """
    if not raw:
        return ""
    return raw.strip().title()


def _cluster_profession(prof_normalised: str) -> str:
    """Map normalised profession → cluster name, fallback to 'other'."""
    key = prof_normalised.strip().lower()
    return _PROFESSION_LOOKUP.get(key, "other")


def _normalise_company_tier(raw: str) -> str | None:
    """Normalise company tier to 'Tier 1' / 'Tier 2' / 'Tier 3'."""
    if not raw:
        return None
    raw_lower = raw.strip().lower()
    for label in ("tier 1", "tier1", "t1", "1"):
        if raw_lower == label:
            return "Tier 1"
    for label in ("tier 2", "tier2", "t2", "2"):
        if raw_lower == label:
            return "Tier 2"
    for label in ("tier 3", "tier3", "t3", "3"):
        if raw_lower == label:
            return "Tier 3"
    return raw   # pass through unknown


def _company_tier_to_int(tier: str | None) -> int | None:
    """Tier 1→1, Tier 2→2, Tier 3→3, else None."""
    _map = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
    return _map.get(tier) if tier else None


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EMAIL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_FREE_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "yahoo.com", "yahoo.in", "yahoo.co.in", "hotmail.com",
    "outlook.com", "rediffmail.com", "rediff.com", "icloud.com",
    "live.com", "msn.com", "protonmail.com", "tutanota.com",
    "zoho.com", "aol.com", "yandex.com", "yandex.ru",
})


def _is_free_email(domain: str) -> bool:
    return domain in _FREE_EMAIL_DOMAINS


def _extract_domain(email: str) -> str | None:
    try:
        return email.strip().split("@")[1].lower()
    except (IndexError, AttributeError):
        return None


def _string_entropy(s: str) -> float | None:
    """Shannon entropy of character distribution — higher = more random."""
    if not s:
        return None
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _count_digits(s: str) -> int:
    return sum(1 for ch in s if ch.isdigit())


def _digit_ratio(s: str) -> float | None:
    if not s:
        return None
    return _count_digits(s) / len(s)


def _has_suspicious_keywords(username: str) -> bool:
    return any(kw in username for kw in _SUSPICIOUS_USERNAME_WORDS)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  BEHAVIORAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO8601 / unix timestamp to timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


def _parse_timestamp_list(value: Any) -> list[datetime]:
    """Parse a list of timestamps (JSON string or Python list) into datetimes."""
    items = _parse_json_field(value, default=[])
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        dt = _parse_datetime(item)
        if dt:
            result.append(dt)
    return sorted(result)


def _parse_string_list(value: Any) -> list[str]:
    """Parse a JSON-encoded list of strings, or a comma-separated string."""
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except (json.JSONDecodeError, ValueError):
            return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _days_between(dt1: datetime | None, dt2: datetime | None) -> float | None:
    """Return absolute day difference, or None if either is missing."""
    if dt1 is None or dt2 is None:
        return None
    return abs((dt2 - dt1).total_seconds() / 86400)


def _timestamp_burstiness(timestamps: list[datetime]) -> float | None:
    """
    Coefficient of variation of inter-event gaps.
    High CV → irregular bursts; low CV → steady pattern.
    Returns None if fewer than 2 events.
    """
    if len(timestamps) < 2:
        return None
    gaps = [
        (timestamps[i + 1] - timestamps[i]).total_seconds()
        for i in range(len(timestamps) - 1)
    ]
    mean_gap = statistics.mean(gaps)
    if mean_gap == 0:
        return 0.0
    stdev_gap = statistics.pstdev(gaps)
    return stdev_gap / mean_gap   # coefficient of variation


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TEXTUAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ngram_repetition_score(words: list[str], n: int = 4) -> float:
    """
    Fraction of n-grams that are duplicated.
    0.0 = no repetition; 1.0 = fully repetitive text.
    """
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    seen:  set[tuple] = set()
    dupes: int = 0
    for ng in ngrams:
        if ng in seen:
            dupes += 1
        seen.add(ng)
    return dupes / len(ngrams)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  GENERIC UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _getter(profile: dict):
    """Return a safe field accessor closure over the profile dict."""
    def get(field: str, default=None):
        return profile.get(field, default)
    return get


def _safe_cast(value: Any, cast_type: type) -> Any:
    """Cast value to cast_type; return None on failure."""
    if value is None:
        return None
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> float | None:
    return _safe_cast(value, float)


def _safe_bool(value: Any) -> bool | None:
    """
    Coerce value to bool. Handles True/False, 1/0, 'true'/'false' strings.
    Returns None for unrecognisable input.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


def _normalise_str(value: Any) -> str | None:
    """Strip and title-case a string; return None for empty/None."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _clamp(value: Any, low: Any, high: Any) -> Any:
    """Clamp numeric value within [low, high]. None bounds = unbounded."""
    if value is None:
        return None
    if low is not None and value < low:
        return low
    if high is not None and value > high:
        return high
    return value


def _parse_json_field(value: Any, default: Any = None) -> Any:
    """
    Safely parse a field that may be a JSON string, a dict, or a list.
    Returns default on parse failure.
    """
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return default
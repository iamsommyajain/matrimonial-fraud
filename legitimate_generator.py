"""
legitimate_generator.py

Generates statistically coherent, internally consistent legitimate profiles.

The key design principle: every field must be conditionally generated FROM
earlier fields, not independently. For example:
  - profession depends on education_level
  - income depends on profession AND years_experience
  - email_domain depends on profession (doctors have hospital emails, etc.)
  - college depends on education_level

This ensures M1 (functional consistency model) has a genuine signal to detect:
  legitimate profiles will have low fuzzy inconsistency scores because
  their fields are coherent, while fraud profiles won't.
"""

import numpy as np
import random
from datetime import datetime, timedelta
from uuid import uuid4

from constants import (
    GENDERS, GENDER_WEIGHTS,
    EDUCATION_LEVELS, EDUCATION_WEIGHTS, EDUCATION_FIELDS, COLLEGES,
    COLLEGE_CATEGORY_WEIGHTS, CITY_METADATA, COLLEGE_METADATA,
    PROFESSIONS_BY_EDUCATION, COMPANIES_LEGITIMATE, COMPANY_METADATA,
    INCOME_BASE_BY_PROFESSION,
)
from bio_templates import BIO_TEMPLATES, PARTNER_PREF_TEMPLATES
from email_features import extract_email_risk_features
from email_generator import generate_legitimate_email
from hobby_generator import generate_hobbies
from regional_demographics import generate_regional_demographics

rng = np.random.default_rng()   # single shared RNG — seed at call site

# ── Additional demographic attributes ─────────────────────────────

MARITAL_STATUS_DIST = {
    "never_married": 0.72,
    "divorced": 0.14,
    "widowed": 0.08,
    "separated": 0.06,
}

DEVICE_TYPE_DIST = {
    "android": 0.68,
    "ios": 0.27,
    "web": 0.05,
}

def _sample_from_dist(dist):
    keys = list(dist.keys())
    probs = list(dist.values())
    return rng.choice(keys, p=probs)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick(pool, weights=None):
    """Weighted random pick from a list."""
    if weights:
        return random.choices(pool, weights=weights, k=1)[0]
    return random.choice(pool)


def _pick_weighted_dict(d):
    keys = list(d.keys())
    weights = list(d.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _normalize(dist):
    total = sum(dist.values())
    return {key: value / total for key, value in dist.items() if value > 0}


def _weighted_indexed_pick(items, weights):
    return random.choices(items, weights=weights, k=1)[0]


def _generate_latent_traits():
    """
    Generate hidden causal traits before visible fields.

    Independent field sampling creates synthetic artifacts: elite college,
    weak language profile, low-skill job, and luxury behavior can accidentally
    co-occur too often. Latent variables introduce realistic shared causes
    while preserving noise and diversity. These traits are internal by default;
    they condition downstream fields but are not exported unless debug mode is
    explicitly requested.
    """
    ses = _pick_weighted_dict({
        "low": 0.12,
        "lower_middle": 0.26,
        "middle": 0.34,
        "upper_middle": 0.22,
        "elite": 0.06,
    })
    ses_score = {
        "low": 0.18,
        "lower_middle": 0.35,
        "middle": 0.52,
        "upper_middle": 0.72,
        "elite": 0.90,
    }[ses]

    urbanity = _clamp(random.gauss(0.35 + 0.45 * ses_score, 0.18))
    english_fluency = _clamp(random.gauss(0.20 + 0.45 * ses_score + 0.25 * urbanity, 0.16))
    traditionalism = _clamp(random.gauss(0.70 - 0.25 * urbanity - 0.18 * english_fluency, 0.18))
    career_orientation = _clamp(random.gauss(0.30 + 0.34 * ses_score + 0.28 * english_fluency, 0.18))
    tech_savviness = _clamp(random.gauss(0.20 + 0.35 * urbanity + 0.25 * english_fluency, 0.18))
    activity_level = _clamp(random.gauss(0.45 + 0.15 * urbanity - 0.10 * traditionalism, 0.20))
    attractiveness_score = _clamp(random.gauss(0.50 + 0.08 * ses_score + 0.05 * activity_level, 0.18))

    return {
        "socioeconomic_tier": ses,
        "ses_score": ses_score,
        "urbanity": urbanity,
        "english_fluency": english_fluency,
        "traditionalism": traditionalism,
        "career_orientation": career_orientation,
        "tech_savviness": tech_savviness,
        "activity_level": activity_level,
        "attractiveness_score": attractiveness_score,
    }


def _generate_geography(gender, latent_traits):
    """
    Geography is a visible consequence of native region plus latent mobility.

    Migration remains probabilistic. High career orientation, higher SES, and
    urban exposure raise migration odds, but local rootedness still appears.
    """
    migration_probability = _clamp(
        0.08
        + 0.12 * latent_traits["career_orientation"]
        + 0.08 * latent_traits["ses_score"]
        + 0.06 * latent_traits["urbanity"]
        - 0.05 * latent_traits["traditionalism"],
        0.05,
        0.38,
    )
    demographics = generate_regional_demographics(
        gender,
        migration=True,
        migration_probability=migration_probability,
    )
    # A small rural hometown path keeps the city model from pretending every
    # profile originates in a named urban center. It is low probability and
    # mainly triggered by low urbanity, preserving backward-compatible strings.
    rural_probability = _clamp(0.16 * (1 - latent_traits["urbanity"]) - 0.05 * latent_traits["ses_score"], 0.0, 0.14)
    if not demographics["migrated"] and random.random() < rural_probability:
        rural_city = f"Rural {demographics['native_state']}"
        if rural_city in CITY_METADATA:
            demographics["city"] = rural_city
            demographics["state"] = demographics["native_state"]
    return demographics


def _pick_college(education_level):
    category = _pick_weighted_dict(COLLEGE_CATEGORY_WEIGHTS[education_level])
    return _pick(COLLEGES[education_level][category])


def _generate_family_background(latent_traits, city):
    """
    Family background is not exported as latents, but it shapes narration.
    """
    style_weights = {
        "traditional": 0.20 + 0.45 * latent_traits["traditionalism"],
        "modern yet rooted": 0.35,
        "close-knit": 0.34,
        "simple": 0.25 + 0.20 * (1 - latent_traits["ses_score"]),
    }
    father_weights = {
        "retired": 0.18,
        "a businessman": 0.16 + 0.20 * latent_traits["ses_score"],
        "a government employee": 0.20 + 0.10 * latent_traits["traditionalism"],
        "a professional": 0.18 + 0.24 * latent_traits["english_fluency"],
    }
    mother_weights = {
        "a homemaker": 0.35 + 0.22 * latent_traits["traditionalism"],
        "a teacher": 0.22,
        "also working": 0.18 + 0.32 * latent_traits["urbanity"],
    }
    return {
        "family_style": _pick_weighted_dict(_normalize(style_weights)),
        "father_role": _pick_weighted_dict(_normalize(father_weights)),
        "mother_role": _pick_weighted_dict(_normalize(mother_weights)),
        "siblings": random.choices([0, 1, 2, 3], weights=[0.12, 0.42, 0.32, 0.14], k=1)[0],
        "city": city,
    }


def _generate_education_profile(latent_traits, city):
    """
    Build an education profile in stages so downstream career and income
    depend on a coherent socioeconomic trajectory instead of independent
    sampling.

    Marginal realism is not enough: the model must preserve joint structure
    between latent SES, city opportunity, field choice, college prestige, and
    eventual profession. This helper produces a school quality score, then
    samples education level, field, college tier, and college name.
    """
    city_meta = CITY_METADATA.get(city, {"tier": "tier3", "education_hub": False})
    school_quality = _clamp(
        0.20 * latent_traits["ses_score"]
        + 0.22 * latent_traits["english_fluency"]
        + 0.18 * latent_traits["career_orientation"]
        + 0.16 * latent_traits["urbanity"]
        + (0.10 if city_meta["education_hub"] else -0.05)
        + random.gauss(0, 0.08)
    )

    level_weights = dict(zip(EDUCATION_LEVELS, EDUCATION_WEIGHTS))
    level_weights["10th"] *= 1.45 - 0.95 * school_quality
    level_weights["12th"] *= 1.25 - 0.70 * school_quality
    level_weights["Graduate"] *= 0.80 + 0.55 * school_quality
    level_weights["Post Graduate"] *= 0.65 + 1.10 * school_quality
    level_weights["PhD"] *= 0.30 + 0.90 * school_quality
    education_level = _pick_weighted_dict(_normalize(level_weights))

    field_weights = {field: 1.0 for field in EDUCATION_FIELDS[education_level]}
    for field in list(field_weights):
        base = 1.0
        flat = field.lower()
        if field in {"Engineering", "Science", "M.Tech", "MCA"}:
            base *= 0.75 + 1.15 * latent_traits["career_orientation"]
        if field in {"MBA", "Management", "Commerce", "M.Com"}:
            base *= 0.60 + 1.20 * latent_traits["ses_score"]
        if field in {"Arts", "Law", "M.A", "M.Sc"}:
            base *= 0.85 + 0.30 * latent_traits["english_fluency"]
        if any(k in flat for k in ["medicine", "md", "ms", "nursing", "pharmacy"]):
            base *= 0.55 + 1.05 * school_quality
        if education_level in {"10th", "12th"} and field == "Commerce":
            base *= 1.10
        if education_level == "PhD" and field in {"Humanities", "Sciences", "Engineering", "Management"}:
            base *= 1.10
        field_weights[field] *= base
    education_field = _pick_weighted_dict(_normalize(field_weights))

    category_weights = dict(COLLEGE_CATEGORY_WEIGHTS[education_level])
    access_score = school_quality + (0.12 if city_meta["tier"] == "tier1" else 0.06 if city_meta["tier"] == "tier2" else -0.05)
    for category in category_weights:
        if category.startswith("elite") or category in {"top_research"}:
            category_weights[category] *= 0.25 + 1.40 * access_score
        elif category in {"top_universities", "technical_pg", "medical_pg", "good_universities"}:
            category_weights[category] *= 0.55 + 1.00 * access_score
        elif category in {"private_reputed", "general_pg"}:
            category_weights[category] *= 0.95 + 0.40 * access_score
        elif category in {"regional_state", "generic"}:
            category_weights[category] *= 1.15 - 0.25 * access_score
        else:
            category_weights[category] *= 1.05 - 0.15 * access_score

    category_choice = _pick_weighted_dict(_normalize(category_weights))
    colleges = COLLEGES[education_level][category_choice]

    college_weights = []
    for college_name in colleges:
        metadata = COLLEGE_METADATA.get(college_name, {})
        weight = metadata.get("acceptance_weight", 0.18)
        if education_field in metadata.get("fields", []):
            weight *= 1.35
        if metadata.get("tier") == "elite":
            weight *= 0.27 + 1.50 * school_quality
        elif metadata.get("tier") == "premium":
            weight *= 0.55 + 1.05 * school_quality
        elif metadata.get("tier") == "upper_mid":
            weight *= 0.90 + 0.35 * school_quality
        elif metadata.get("tier") == "mid":
            weight *= 1.05 - 0.15 * (1 - school_quality)
        if city_meta["tier"] == "tier1" and metadata.get("migration_affinity", 0.0) > 0.4:
            weight *= 1.12
        college_weights.append(max(0.002, weight))

    college_name = _weighted_indexed_pick(colleges, college_weights)
    college_metadata = COLLEGE_METADATA.get(college_name, {})
    college_tier = college_metadata.get("tier", "local")

    return {
        "education_level": education_level,
        "education_field": education_field,
        "college_name": college_name,
        "college": college_name,
        "college_tier": college_tier,
        "college_prestige": college_tier,
        "school_quality": school_quality,
        "education_quality_score": _clamp(school_quality),
    }


def _generate_profession_profile(age, city, education_profile, latent_traits):
    education_level = education_profile["education_level"]
    education_field = education_profile["education_field"]
    prestige = education_profile["college_tier"]
    school_quality = education_profile.get("school_quality", 0.5)
    city_meta = CITY_METADATA.get(city, {"tier": "tier3", "tech_hub": False})

    profession_weights = {profession: 1.0 for profession in PROFESSIONS_BY_EDUCATION[education_level]}
    for profession in profession_weights:
        p = profession.lower()
        if any(k in p for k in ["software", "data scientist", "product manager", "principal engineer"]):
            profession_weights[profession] *= 0.55 + 1.40 * latent_traits["tech_savviness"]
        if any(k in p for k in ["doctor", "lawyer", "chartered accountant", "ias", "ips", "research", "professor"]):
            profession_weights[profession] *= 0.55 + 1.20 * latent_traits["career_orientation"]
        if any(k in p for k in ["business", "shop owner", "self employed", "family business"]):
            profession_weights[profession] *= 0.70 + 0.85 * latent_traits["ses_score"]
        if education_level in {"10th", "12th"} and profession in PROFESSIONS_BY_EDUCATION[education_level]:
            profession_weights[profession] *= 1.35
        if education_level in {"10th", "12th"} and profession not in PROFESSIONS_BY_EDUCATION[education_level]:
            profession_weights[profession] *= 0.2

        if education_field in {"Engineering", "Science"} and any(k in p for k in ["software", "data scientist", "principal engineer", "civil engineer", "research scientist"]):
            profession_weights[profession] *= 1.45
        if education_field in {"Management", "MBA", "Commerce"} and any(k in p for k in ["product manager", "finance manager", "consultant", "chartered accountant", "bank officer", "business analyst"]):
            profession_weights[profession] *= 1.45
        if education_field == "Medicine" and any(k in p for k in ["doctor", "pharmacist", "nurse"]):
            profession_weights[profession] *= 1.60
        if education_field == "Law" and any(k in p for k in ["lawyer", "government ias", "government ips"]):
            profession_weights[profession] *= 1.45
        if education_field == "Arts" and any(k in p for k in ["teacher", "professor", "marketing executive", "hr executive", "government employee"]):
            profession_weights[profession] *= 1.25
        if education_field == "Science" and any(k in p for k in ["research scientist", "professor", "government employee"]):
            profession_weights[profession] *= 1.20

        if prestige == "elite" and any(k in p for k in ["software", "product", "data", "research", "principal", "consultant"]):
            profession_weights[profession] *= 1.75
        elif prestige == "premium" and any(k in p for k in ["product", "consultant", "senior software", "finance manager"]):
            profession_weights[profession] *= 1.30
        elif prestige == "local" and any(k in p for k in ["shop owner", "driver", "sales executive", "office assistant", "data entry operator"]):
            profession_weights[profession] *= 1.35

        if city_meta["tech_hub"] and any(k in p for k in ["software", "data scientist", "product manager", "principal engineer"]):
            profession_weights[profession] *= 1.15
        if city_meta["tier"] == "tier3" and any(k in p for k in ["shop owner", "driver", "sales executive", "delivery executive"]):
            profession_weights[profession] *= 1.2
        if city_meta["tier"] == "tier1" and any(k in p for k in ["finance manager", "consultant", "product manager", "data scientist"]):
            profession_weights[profession] *= 1.15

    profession = _pick_weighted_dict(_normalize(profession_weights))
    company_name, company_tier = _pick_company_for_profession(profession, city, education_profile, latent_traits)
    company_metadata = COMPANY_METADATA.get(company_name, {
        "tier": company_tier,
        "industry": "business",
        "salary_multiplier": 1.0,
        "city_affinity": [city_meta.get("tier", "tier3")],
    })
    years_experience = _generate_experience(age, education_profile, latent_traits)
    annual_income_lpa = _realistic_income(
        profession,
        company_metadata,
        education_profile,
        city,
        latent_traits,
        years_experience,
    )

    return {
        "profession": profession,
        "company_name": company_name,
        "company": company_name,
        "company_tier": company_tier,
        "years_experience": years_experience,
        "annual_income_lpa": annual_income_lpa,
    }


def _pick_company_for_profession(profession, city, education_profile, latent_traits):
    city_meta = CITY_METADATA.get(city, {"tier": "tier3", "tech_hub": False})
    profession_lower = profession.lower()
    if any(k in profession_lower for k in ["software", "data scientist", "product manager", "principal engineer"]):
        industry = "tech"
    elif any(k in profession_lower for k in ["doctor", "nurse", "pharmacist"]):
        industry = "medical"
    elif any(k in profession_lower for k in ["bank officer", "chartered accountant", "finance manager", "accountant"]):
        industry = "finance"
    elif any(k in profession_lower for k in ["consultant", "business analyst", "marketing executive", "hr executive", "lawyer"]):
        industry = "consulting"
    elif any(k in profession_lower for k in ["government", "ias", "ips", "police", "railways", "army"]):
        industry = "government"
    elif any(k in profession_lower for k in ["teacher", "professor"]):
        industry = "education"
    elif any(k in profession_lower for k in ["shop owner", "sales executive", "delivery executive", "office assistant", "data entry operator", "driver"]):
        industry = "business"
    else:
        industry = "business"

    candidates = list(COMPANIES_LEGITIMATE.get(industry, COMPANIES_LEGITIMATE["business"]))
    if industry == "tech" and education_profile["college_tier"] == "elite":
        candidates += [company for company in COMPANY_METADATA if COMPANY_METADATA[company]["industry"] == "tech" and COMPANY_METADATA[company]["tier"] == "elite"]

    company_weights = []
    for company in candidates:
        meta = COMPANY_METADATA.get(company, {
            "tier": "local",
            "industry": industry,
            "salary_multiplier": 1.0,
            "city_affinity": [city_meta.get("tier", "tier3")],
        })
        weight = 1.0
        if meta["industry"] == industry:
            weight *= 1.2
        if city_meta["tier"] in meta.get("city_affinity", []):
            weight *= 1.15
        if education_profile["college_tier"] == "elite" and meta["tier"] == "elite":
            weight *= 1.9
        if education_profile["college_tier"] == "local" and meta["tier"] == "elite":
            weight *= 0.08
        if industry == "government" and meta["industry"] == "government":
            weight *= 1.4
        if industry == "medical" and meta["industry"] == "medical":
            weight *= 1.3
        if industry == "tech" and city_meta["tech_hub"] and meta["tier"] in {"elite", "upper_mid"}:
            weight *= 1.2
        if industry == "business" and education_profile["college_tier"] in {"local", "mid"} and meta["tier"] in {"local", "mid"}:
            weight *= 1.25
        if profession_lower.startswith("consultant") and meta["industry"] == "consulting":
            weight *= 1.2
        company_weights.append(max(0.01, weight))

    if not candidates:
        return "Family Business", "self_employed"

    company_name = _weighted_indexed_pick(candidates, company_weights)
    company_meta = COMPANY_METADATA.get(company_name, {})
    company_tier = company_meta.get("tier", "local")

    if random.random() < 0.04 and company_tier in {"mid", "local"}:
        generic_names = ["Private Consultancy", "Regional Services", "Family Business", "Local Trading Co."]
        company_name = random.choice(generic_names)
        company_tier = "local"

    return company_name, company_tier


def _generate_device_type(age, profession, latent_traits):
    device_dist = dict(DEVICE_TYPE_DIST)
    device_dist["ios"] *= 0.65 + 1.35 * latent_traits["ses_score"]
    device_dist["web"] *= 0.75 + 1.20 * latent_traits["tech_savviness"]
    device_dist["android"] *= 1.15 - 0.35 * latent_traits["ses_score"]
    if age < 30:
        device_dist["ios"] *= 1.20
    if any(k in profession.lower() for k in ["doctor", "surgeon", "product manager"]):
        device_dist["ios"] *= 1.25
    return _sample_from_dist(_normalize(device_dist))


def _generate_lifestyle_profile(name, age, gender, city, marital_status, education_profile, profession_profile, family_background, latent_traits):
    bio_text = _generate_bio(name, profession_profile["profession"], city, gender)
    if latent_traits["english_fluency"] < 0.30 and random.random() < 0.20:
        bio_text = bio_text.replace("looking for", "looking for a").replace("well-settled", "settled")

    partner_prefs = _generate_partner_prefs(age, city)
    if latent_traits["traditionalism"] > 0.68 and random.random() < 0.35:
        partner_prefs += " Family values and cultural compatibility are important."
    elif latent_traits["career_orientation"] > 0.70 and random.random() < 0.30:
        partner_prefs += " Prefer someone supportive of professional growth."

    about_family = (
        f"We are a {family_background['family_style']} family based in {city}. "
        f"My father is {family_background['father_role']} and my mother is {family_background['mother_role']}. "
        f"We have {family_background['siblings']} sibling(s) and share a very warm household."
    )

    hobbies = generate_hobbies(
        age=age,
        education_level=education_profile["education_level"],
        profession=profession_profile["profession"],
        city=city,
        income=profession_profile["annual_income_lpa"],
        marital_status=marital_status,
    )
    return {
        "bio_text": bio_text,
        "partner_preferences": partner_prefs,
        "about_family": about_family,
        "hobbies": hobbies,
    }


def _generate_behavioral_profile(created_at, latent_traits, age, profession):
    timestamps = _generate_timestamps(created_at)
    base_messages = 30 if latent_traits["tech_savviness"] > 0.65 else 22
    activity_multiplier = 0.55 + latent_traits["activity_level"]
    messages_sent = random.randint(0, max(3, int(base_messages * activity_multiplier)))
    messages_received = int(messages_sent * random.uniform(0.3, 1.5))
    match_requests = random.randint(0, max(2, int(15 * activity_multiplier)))
    match_accepts = int(match_requests * random.uniform(0.1, 0.5))
    unique_contacts = random.randint(1, min(20, messages_sent + 1))
    device_type = _generate_device_type(age, profession, latent_traits)
    return {
        "timestamps": timestamps,
        "device_type": device_type,
        "messages_sent": messages_sent,
        "messages_received": messages_received,
        "match_requests": match_requests,
        "match_accepts": match_accepts,
        "unique_contacts": unique_contacts,
    }


def _validate_profile_consistency(profile):
    """
    Lightweight debug validator. It warns about impossible combinations but
    never raises; synthetic population generation should tolerate rare edges.
    """
    warnings = []
    elite_colleges = {
        name for name, meta in COLLEGE_METADATA.items()
        if meta.get("tier") == "elite"
    }
    if profile.get("college_name") in elite_colleges and profile.get("education_level") in ("10th", "12th"):
        warnings.append("elite_college_with_school_level")
    if profile.get("college_name") in elite_colleges and profile.get("profession") in ("Daily Wage Worker", "Domestic Worker", "Security Guard"):
        warnings.append("elite_college_low_skill_profession")
    if profile.get("company_name") in {"Google India", "Microsoft India", "Meta"}:
        city_meta = CITY_METADATA.get(profile.get("city"), {})
        if city_meta.get("tier") == "tier3" and not profile.get("is_migrant"):
            warnings.append("tier3_non_migrant_global_tech_company")
    if (
        profile.get("annual_income_lpa", 0) > 35
        and profile.get("years_experience", 0) < 3
        and profile.get("profession") in {"Shop Owner", "Driver", "Office Assistant", "Data Entry Operator"}
    ):
        warnings.append("very_high_income_low_skill_low_experience")

    if (
        profile.get("college_name") in elite_colleges
        and profile.get("company_name") in {"Family Business", "Private Consultancy", "Regional Services", "Local Trading Co."}
        and profile.get("annual_income_lpa", 0) < 5
    ):
        warnings.append("elite_college_low_income_local_company")

    if profile.get("years_experience", 0) <= 2 and profile.get("annual_income_lpa", 0) > 28:
        warnings.append("very_high_income_low_experience")

    if profile.get("education_level") in {"10th", "12th"} and profile.get("company_name") in {"Google India", "Microsoft India", "Meta", "Amazon India"}:
        warnings.append("low_education_elite_company")

    medical_colleges = {
        name for name, meta in COLLEGE_METADATA.items()
        if "Medicine" in meta.get("fields", []) or name.startswith("AIIMS")
    }
    if profile.get("college_name") in medical_colleges and profile.get("profession") not in {"Doctor", "Nurse", "Pharmacist", "Government Employee", "Professor"}:
        warnings.append("medical_college_unrelated_profession")

    return warnings


def _generate_experience(age, education_profile, latent_traits):
    """
    Simulate realistic work experience as the result of education delay,
    career orientation, family orientation, and occasional gaps.

    PhD holders enter later and still can accumulate meaningful experience if
    they remain career-oriented. Traditional or low-career-orientation profiles
    are more likely to have delayed starts or interrupted early careers.
    """
    education_level = education_profile["education_level"]
    min_work_age = {
        "10th": 16,
        "12th": 18,
        "Graduate": 22,
        "Post Graduate": 24,
        "PhD": 28,
    }[education_level]

    base_delay = 0
    if education_level == "PhD":
        base_delay += random.randint(1, 3)
    elif education_level == "Post Graduate":
        base_delay += random.choice([0, 1, 1, 2])
    elif education_level == "Graduate":
        base_delay += random.choice([0, 0, 1, 1, 2])
    else:
        base_delay += random.choice([0, 1, 1, 2])

    if latent_traits["career_orientation"] < 0.35:
        base_delay += random.choice([0, 1, 2])
    if latent_traits["traditionalism"] > 0.70 and random.random() < 0.22:
        base_delay += random.randint(1, 2)
    if latent_traits["ses_score"] > 0.78 and education_level in {"Graduate", "Post Graduate", "PhD"}:
        base_delay = max(0, base_delay - 1)

    max_possible = max(0, age - min_work_age)
    expected_experience = max(0, max_possible - base_delay)
    years_experience = int(round(_clamp(np.random.normal(expected_experience, 1.2), 0, max_possible)))

    if years_experience > 2 and random.random() < 0.15:
        years_experience = max(0, years_experience - random.randint(1, 2))

    return years_experience


def _realistic_income(profession, company_profile, education_profile, city, latent_traits, years_experience):
    """
    Generate income as a skewed distribution that reflects profession,
    company prestige, geography, experience, and latent SES.
    """
    base_info = INCOME_BASE_BY_PROFESSION.get(profession, INCOME_BASE_BY_PROFESSION["DEFAULT"])
    base_income = max(1.5, base_info["base"] + base_info["per_year_exp"] * years_experience)

    company_multiplier = company_profile.get("salary_multiplier", 1.0)
    company_tier = company_profile.get("tier", "local")
    education_quality = education_profile.get("school_quality", 0.5)
    city_meta = CITY_METADATA.get(city, {"tier": "tier3"})

    expected = base_income
    expected *= 1.0 + 0.10 * (education_quality - 0.5)
    expected *= company_multiplier
    expected *= 1.0 + (0.14 if city_meta["tier"] == "tier1" else 0.08 if city_meta["tier"] == "tier2" else 0.0)
    expected *= 0.92 + 0.20 * latent_traits["ses_score"] + 0.08 * latent_traits["career_orientation"]

    if company_tier == "elite":
        expected *= 1.22
    elif company_tier == "premium":
        expected *= 1.12
    elif company_tier == "upper_mid":
        expected *= 1.06

    sigma = 0.35 + 0.08 * (base_info["sigma"] / 3.0)
    if any(k in profession.lower() for k in ["software", "data scientist", "product", "principal"]):
        sigma += 0.08
    if company_tier == "self_employed":
        sigma += 0.20
        expected *= 1.08 + random.random() * 0.28
    if profession == "Government Employee":
        sigma = min(sigma, 0.35)
        expected *= 0.95

    expected = max(1.5, expected)
    mu = np.log(max(0.8, expected)) - 0.5 * sigma * sigma
    income = float(np.random.lognormal(mu, sigma))

    if company_tier == "self_employed" and random.random() < 0.10:
        income *= random.uniform(1.1, 1.8)
    if random.random() < 0.04 and years_experience < 3:
        income *= random.uniform(0.75, 0.95)
    if random.random() < 0.02:
        income *= random.uniform(1.05, 1.28)
    if profession == "Government Employee":
        income = min(income, expected * 1.3)

    return max(1.5, round(income, 1))


def _pick_company(profession):
    """
    Pick a plausible company name based on profession category.
    Tech → TCS/Infosys etc. Finance → HDFC/ICICI etc.
    """
    p = profession.lower()
    if any(k in p for k in ["software", "engineer", "developer", "data scientist", "product"]):
        return _pick(COMPANIES_LEGITIMATE["tech"])
    elif any(k in p for k in ["bank", "finance", "accountant", "chartered"]):
        return _pick(COMPANIES_LEGITIMATE["finance"])
    elif any(k in p for k in ["government", "ias", "ips", "police", "army"]):
        return _pick(COMPANIES_LEGITIMATE["govt"])
    elif any(k in p for k in ["doctor", "physician", "surgeon", "medical"]):
        return _pick(COMPANIES_LEGITIMATE["medical"])
    elif any(k in p for k in ["professor", "teacher", "lecturer"]):
        return _pick(COMPANIES_LEGITIMATE["education"])
    elif any(k in p for k in ["business", "owner", "self", "entrepreneur"]):
        return _pick(COMPANIES_LEGITIMATE["business"])
    else:
        # Mix of tech and general
        return _pick(COMPANIES_LEGITIMATE["tech"] + COMPANIES_LEGITIMATE["finance"])


def _generate_bio(name, profession, city, gender):
    """
    Pick a bio template (filtered by gender), fill slots, and clean up.
    Legitimate profiles get a single, coherent bio.
    """
    compatible = [b for b in BIO_TEMPLATES
                  if b["gender"] in ("Any", gender)]
    template = random.choice(compatible)["text"]

    bio = template.replace("{profession}", profession)
    bio = bio.replace("{city}", city)
    # Clean the son/daughter placeholder
    if gender == "Male":
        bio = bio.replace("son/daughter", "son")
        bio = bio.replace("her/his", "his")
        bio = bio.replace("his/her", "his")
    else:
        bio = bio.replace("son/daughter", "daughter")
        bio = bio.replace("her/his", "her")
        bio = bio.replace("his/her", "her")
    return bio


def _generate_partner_prefs(age, city):
    """Generate a partner preference string."""
    template = random.choice(PARTNER_PREF_TEMPLATES)
    age_min = max(21, age - 5)
    age_max = age + 5
    height_pref = random.randint(155, 175)
    return (template
            .replace("{age_min}", str(age_min))
            .replace("{age_max}", str(age_max))
            .replace("{city}", city)
            .replace("{height_pref}", str(height_pref)))


def _generate_timestamps(created_at, is_active=True):
    """
    Simulate realistic login and edit timestamps for a profile.

    Legitimate users:
      - Sporadic logins (2-4 per week on average)
      - Login IPs mostly from one city, occasionally travel
      - Edit profile 1-3 times in first week, rarely after
      - Photos uploaded over multiple days
    """
    logins = []
    edits = []
    photo_dates = []

    days_since_creation = (datetime.now() - created_at).days
    active_days = min(days_since_creation, 60)  # look at last 60 days

    # Login timestamps: Poisson-distributed, ~3 per week
    for day_offset in range(active_days):
        n_logins_today = np.random.poisson(0.4)  # avg 3/week ≈ 0.43/day
        for _ in range(n_logins_today):
            hour = random.randint(8, 23)
            minute = random.randint(0, 59)
            logins.append(created_at + timedelta(days=day_offset, hours=hour, minutes=minute))

    # Edit timestamps: clustered in first week
    n_edits = random.randint(1, 4)
    for _ in range(n_edits):
        edit_day = random.randint(0, min(7, active_days))
        edits.append(created_at + timedelta(
            days=edit_day, hours=random.randint(9, 22), minutes=random.randint(0, 59)
        ))

    # Photo upload dates: spread over first 3 days
    n_photos = random.randint(2, 6)
    for i in range(n_photos):
        upload_day = random.randint(0, min(3, active_days))
        photo_dates.append(created_at + timedelta(
            days=upload_day, hours=random.randint(10, 21), minutes=random.randint(0, 59)
        ))

    # Login IPs: mostly one region, slight variation (legitimate travel)
    city_ip_prefix = f"103.{random.randint(1, 254)}.{random.randint(1, 254)}"
    login_ips = []
    for _ in logins:
        if random.random() < 0.05:  # 5% travel / VPN
            login_ips.append(f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}")
        else:
            login_ips.append(f"{city_ip_prefix}.{random.randint(1, 254)}")

    return {
        "login_timestamps": [t.isoformat() for t in sorted(logins)],
        "edit_timestamps": [t.isoformat() for t in sorted(edits)],
        "photo_upload_dates": [t.isoformat() for t in sorted(photo_dates)],
        "login_ip_list": login_ips,
    }


def _mock_face_embedding():
    """
    Return a mock 128-dim face embedding vector sampled from a Gaussian.
    In real M3 implementation this comes from a FaceNet/ArcFace model.
    We use a unique per-profile embedding so fraud injection (sharing
    embeddings across profiles) creates a detectable signal.
    """
    return np.random.normal(0, 1, 128).tolist()


def _mock_exif(is_legitimate=True):
    """
    Return a plausible EXIF metadata dict.
    Legitimate: device info present, software is camera OS.
    Fraud (called from fraud generator): stripped or suspicious.
    """
    if is_legitimate:
        devices = ["Apple iPhone 14", "Samsung Galaxy S22", "OnePlus 11",
                   "Xiaomi 13 Pro", "Google Pixel 7", "Apple iPhone 13"]
        return {
            "device": random.choice(devices),
            "software": "Camera App",
            "gps_stripped": random.choice([True, False]),  # some users strip for privacy
            "timestamp_consistent": True,
            "source_suspicious": False,
        }
    else:
        return {
            "device": None,
            "software": random.choice(["GIMP 2.10", "Adobe Photoshop", "Unknown", None]),
            "gps_stripped": True,
            "timestamp_consistent": False,
            "source_suspicious": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def generate_legitimate_profile(created_at=None, include_debug_latents=False):
    """
    Generate one legitimate matrimonial profile from a causal population model.

    Causal order:
      latent_traits -> geography/family -> education -> profession/income
      -> lifestyle/content -> behavioral signals.

    Latents create shared causes across fields, which helps downstream fraud
    models learn meaningful inconsistency signals instead of shallow artifacts
    from independently sampled columns. The returned schema stays backward
    compatible; hidden latents are included only when include_debug_latents=True.
    """
    if created_at is None:
        # Profiles created anywhere in the last 6 months
        created_at = datetime.now() - timedelta(days=random.randint(1, 180))

    latent_traits = _generate_latent_traits()

    # ── Step 1: Demographics and geography from latent mobility ────────────
    gender = _pick(GENDERS, GENDER_WEIGHTS)
    demographics = _generate_geography(gender, latent_traits)
    state = demographics["state"]
    city = demographics["city"]
    native_state = demographics["native_state"]
    is_migrant = demographics["migrated"]
    religion = demographics["religion"]
    caste = demographics["caste"]
    mother_tongue = demographics["mother_tongue"]
    name = demographics["name"]
    age = int(round(random.triangular(22, 50, 30 + 8 * (1 - latent_traits["career_orientation"]))))

    # ── Marital status conditioned on age ───────────────────────────
    if age <= 26:
        marital_dist = {
            "never_married": 0.94,
            "divorced": 0.03,
            "widowed": 0.01,
            "separated": 0.02,
        }
    elif age <= 35:
        marital_dist = {
            "never_married": 0.78,
            "divorced": 0.12,
            "widowed": 0.03,
            "separated": 0.07,
        }
    else:
        marital_dist = {
            "never_married": 0.45,
            "divorced": 0.30,
            "widowed": 0.15,
            "separated": 0.10,
        }

    marital_status = _sample_from_dist(marital_dist)
    family_background = _generate_family_background(latent_traits, city)

    height_cm    = (
        random.randint(165, 188) if gender == "Male"
        else random.randint(152, 172)
    )

    # Keep geography internally coherent
    country = "India"

    # ── Step 2-4: Education -> profession -> income causal chain ───────────
    education_profile = _generate_education_profile(latent_traits, city)
    education_level = education_profile["education_level"]
    education_field = education_profile["education_field"]
    college = education_profile["college_name"]

    profession_profile = _generate_profession_profile(age, city, education_profile, latent_traits)
    profession = profession_profile["profession"]
    company = profession_profile["company"]
    years_experience = profession_profile["years_experience"]
    annual_income_lpa = profession_profile["annual_income_lpa"]

    # ── Step 5: Contact details (conditional on profession) ───────────────
    email = generate_legitimate_email(
        name=name,
        age=age,
        profession=profession,
        company=company,
        education=education_level,
    )
    email_risk_features = extract_email_risk_features(email, {
        "name": name,
        "profession": profession,
        "company_name": company,
        "education_level": education_level,
    })

    # ── Step 6: Lifestyle/content from family, career, and traits ──────────
    lifestyle_profile = _generate_lifestyle_profile(
        name=name,
        age=age,
        gender=gender,
        city=city,
        marital_status=marital_status,
        education_profile=education_profile,
        profession_profile=profession_profile,
        family_background=family_background,
        latent_traits=latent_traits,
    )
    bio_text = lifestyle_profile["bio_text"]
    partner_prefs = lifestyle_profile["partner_preferences"]
    about_family = lifestyle_profile["about_family"]
    hobbies = lifestyle_profile["hobbies"]

    # ── Step 7: Behavioral / temporal signals ─────────────────────────────
    behavioral_profile = _generate_behavioral_profile(created_at, latent_traits, age, profession)
    timestamps = behavioral_profile["timestamps"]
    device_type = behavioral_profile["device_type"]

    # ── Step 8: Visual signals (mocked) ───────────────────────────────────
    n_photos = random.randint(2, 6)
    face_embeddings = [_mock_face_embedding() for _ in range(n_photos)]
    # Legitimate: all embeddings are from same person → should cluster tightly
    # We model this by generating one base embedding and adding tiny noise
    base_embedding = _mock_face_embedding()
    face_embeddings = [
        (np.array(base_embedding) + np.random.normal(0, 0.05, 128)).tolist()
        for _ in range(n_photos)
    ]
    exif_data = [_mock_exif(is_legitimate=True) for _ in range(n_photos)]

    # ── Step 9: Interaction signals (seeded — graph built separately) ──────
    # These remain per-profile stats; the full graph is assembled elsewhere.
    messages_sent = behavioral_profile["messages_sent"]
    messages_received = behavioral_profile["messages_received"]
    match_requests = behavioral_profile["match_requests"]
    match_accepts = behavioral_profile["match_accepts"]
    unique_contacts = behavioral_profile["unique_contacts"]

    # ── Assemble ─────────────────────────────────────────────────────────
    profile = {
        # Identity
        "profile_id":           str(uuid4()),
        "name":                 name,
        "age":                  age,
        "gender":               gender,
        "religion":             religion,
        "caste":                caste,
        "mother_tongue":        mother_tongue,
        "height_cm":            height_cm,
        "city":                 city,
        "state":                state,
        "native_state":         native_state,
        "country":              country,
        "is_migrant":           is_migrant,
        "marital_status": marital_status,
        "device_type": device_type,

        # Education
        "education_level":      education_level,
        "education_field":      education_field,
        "college_name":         college,
        "college_tier":         education_profile["college_tier"],

        # Profession
        "profession":           profession,
        "company_name":         company,
        "company_tier":         profession_profile["company_tier"],
        "years_experience":     years_experience,
        "annual_income_lpa":    annual_income_lpa,
        "email":                email,
        "email_domain":         email.split("@")[1],
        "email_risk_features":  email_risk_features,
        "email_consistency_score": email_risk_features["email_consistency_score"],

        # Bio
        "bio_text":             bio_text,
        "partner_preferences":  partner_prefs,
        "about_family":         about_family,
        "hobbies":              hobbies,

        # Behavioral
        "created_at":           created_at.isoformat(),
        "login_timestamps":     timestamps["login_timestamps"],
        "edit_timestamps":      timestamps["edit_timestamps"],
        "photo_upload_dates":   timestamps["photo_upload_dates"],
        "login_ip_list":        timestamps["login_ip_list"],
        "profile_edit_count":   len(timestamps["edit_timestamps"]),

        # Visual
        "n_photos":             n_photos,
        "face_embeddings":      face_embeddings,
        "exif_data":            exif_data,

        # Interaction
        "messages_sent":        messages_sent,
        "messages_received":    messages_received,
        "match_requests_sent":  match_requests,
        "match_accepts":        match_accepts,
        "unique_contacts":      unique_contacts,

        # Labels
        "is_fraud":             False,
        "fraud_type":           None,
        "fraud_severity":       None,
        "injected_signals":     [],
    }

    validation_warnings = _validate_profile_consistency(profile)
    if include_debug_latents:
        profile["_latent_traits"] = latent_traits
        profile["_family_background"] = family_background
        profile["_education_quality_score"] = education_profile["education_quality_score"]
        profile["_college_prestige"] = education_profile["college_prestige"]
        profile["_validation_warnings"] = validation_warnings

    return profile

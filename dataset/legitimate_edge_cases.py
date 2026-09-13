"""Rare but valid legitimate profiles for negative-control evaluation."""

import random
import re

from .email_features import extract_email_risk_features
from .email_generator import generate_legitimate_email


EDGE_CASE_WEIGHTS = {
    "high_income_nonelite_education": 0.025,
    "older_career_change": 0.020,
    "rural_technology_worker": 0.025,
    "doctor_free_email": 0.020,
    "unusual_hobbies": 0.015,
    "complete_previous_marriage": 0.030,
}

FREE_EMAIL_DOMAINS = (
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
)

UNUSUAL_HOBBY_SETS = (
    ("Classical music", "Astronomy", "Gardening"),
    ("Gaming", "Cooking", "Bird watching"),
    ("Chess", "Trekking", "Pottery"),
    ("Photography", "Calligraphy", "Bird watching"),
)


def sample_legitimate_edge_case() -> str | None:
    """Sample a rare valid exception, or return None for a normal profile."""
    scenarios = list(EDGE_CASE_WEIGHTS)
    weights = list(EDGE_CASE_WEIGHTS.values())
    return random.choices([None, *scenarios], weights=[1.0, *weights], k=1)[0]


def apply_legitimate_edge_case(profile: dict, scenario: str | None) -> dict:
    """Apply one bounded exception while keeping the profile legitimate."""
    if scenario is None:
        return profile

    handlers = {
        "high_income_nonelite_education": _high_income_nonelite_education,
        "older_career_change": _older_career_change,
        "rural_technology_worker": _rural_technology_worker,
        "doctor_free_email": _doctor_free_email,
        "unusual_hobbies": _unusual_hobbies,
        "complete_previous_marriage": _complete_previous_marriage,
    }
    handler = handlers.get(scenario)
    if handler is None:
        raise ValueError(f"Unknown legitimate edge-case scenario: {scenario}")

    handler(profile)
    profile["is_fraud"] = False
    profile["fraud_type"] = None
    profile["fraud_severity"] = None
    profile["injected_signals"] = []
    profile["_legitimate_edge_case"] = scenario
    return profile


def _high_income_nonelite_education(profile: dict) -> None:
    profile["education_level"] = random.choice(["Graduate", "Diploma"])
    profile["education_field"] = random.choice(["Engineering", "Commerce", "Management"])
    profile["college_name"] = random.choice(["Regional College", "State College", "Local Institute"])
    profile["college_tier"] = random.choice(["local", "mid"])
    profile["years_experience"] = random.randint(8, 18)
    profile["annual_income_lpa"] = round(random.uniform(18.0, 45.0), 1)
    profile["bio_text"] += " I built my career through practical experience and steady growth."


def _older_career_change(profile: dict) -> None:
    profile["age"] = random.randint(38, 52)
    profile["education_level"] = random.choice(["Post Graduate", "PhD"])
    profile["profession"] = "Student"
    profile["company_name"] = "Independent Study"
    profile["company_tier"] = "local"
    profile["years_experience"] = random.randint(8, 22)
    profile["annual_income_lpa"] = round(random.uniform(1.5, 4.0), 1)
    _refresh_email(profile)
    profile["career_transition"] = True
    profile["previous_profession"] = random.choice([
        "Bank Employee",
        "Teacher",
        "Sales Manager",
        "Business Owner",
    ])
    profile["bio_text"] += " Currently pursuing a new academic direction after a career change."


def _rural_technology_worker(profile: dict) -> None:
    state = profile.get("state") or profile.get("native_state") or "India"
    profile["city"] = f"Rural {state}"
    profile["profession"] = random.choice([
        "Software Developer",
        "Data Analyst",
        "Technical Support Engineer",
    ])
    profile["company_name"] = random.choice([
        "TCS",
        "Infosys",
        "Wipro",
        "Remote Technology Services",
    ])
    profile["company_tier"] = random.choice(["upper_mid", "premium"])
    profile["is_migrant"] = False
    profile["work_mode"] = "remote"
    _refresh_email(profile)
    profile["bio_text"] += " I work remotely in technology while staying connected to my hometown."


def _doctor_free_email(profile: dict) -> None:
    profile["profession"] = "Doctor"
    profile["education_level"] = "Professional Graduate"
    profile["education_field"] = "Medicine"
    profile["college_name"] = random.choice(["Medical College", "Regional Medical College", "State Medical College"])
    profile["college_tier"] = random.choice(["premium", "upper_mid"])
    profile["company_name"] = random.choice(["Apollo Hospitals", "Fortis Healthcare", "AIIMS"])
    profile["email"] = _free_email(profile.get("name", "user"), profile.get("age"))
    profile["email_domain"] = profile["email"].split("@", 1)[1]
    risk_features = extract_email_risk_features(profile["email"], {
        "name": profile.get("name"),
        "profession": profile.get("profession"),
        "company_name": profile.get("company_name"),
        "education_level": profile.get("education_level"),
    })
    profile["email_risk_features"] = risk_features
    profile["email_consistency_score"] = risk_features["email_consistency_score"]


def _unusual_hobbies(profile: dict) -> None:
    profile["hobbies"] = list(random.choice(UNUSUAL_HOBBY_SETS))


def _complete_previous_marriage(profile: dict) -> None:
    profile["marital_status"] = random.choice(["divorced", "widowed"])
    profile["bio_text"] += " I value honesty, family, and meaningful companionship. Open to a new chapter in life."


def _free_email(name: str, age: int | None) -> str:
    parts = [re.sub(r"[^a-z0-9]", "", part.lower()) for part in name.split()]
    parts = [part for part in parts if part] or ["user"]
    username = parts[0]
    if len(parts) > 1:
        username = f"{username}.{parts[-1]}"
    if age and random.random() < 0.35:
        username += str(2026 - int(age))
    return f"{username}@{random.choice(FREE_EMAIL_DOMAINS)}"


def _refresh_email(profile: dict) -> None:
    email = generate_legitimate_email(
        name=profile.get("name", "user"),
        age=profile.get("age"),
        profession=profile.get("profession"),
        company=profile.get("company_name"),
        education=profile.get("education_level"),
    )
    profile["email"] = email
    profile["email_domain"] = email.split("@", 1)[1]
    risk_features = extract_email_risk_features(email, profile)
    profile["email_risk_features"] = risk_features
    profile["email_consistency_score"] = risk_features["email_consistency_score"]

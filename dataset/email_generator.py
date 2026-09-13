"""
Probabilistic email generation for legitimate and fraud profiles.

Most legitimate users still use personal mail. Institutional mail is a soft
identity signal, more common for doctors, academics, researchers, and government
roles. Fraud emails mix normal personal addresses with rarer impersonation,
typo-squatting, disposable, and fake-corporate patterns.
"""

import random
import re

from .constants import COMPANY_DOMAIN_MAP, EMAIL_DOMAIN_CATEGORIES
from .email_features import domains_for_company

# Fraud email generation patterns and personas
FRAUD_PERSONAS = {
    "low_signal_social": 0.45,           # mostly normal-looking, subtle signals
    "medium_signal_impersonator": 0.35,  # professional impersonation attempts
    "high_risk_disposable": 0.20,        # obvious fraud patterns
}

SUBTLE_PROFESSIONAL_TOKENS = [
    "hr", "careers", "recruit", "admin", "support",
    "billing", "finance", "compliance", "chief", "director",
    "officer", "verified", "official", "secure", "account",
]

MILD_SUSPICIOUS_PATTERNS = [
    "{first}.{last}.{token}",
    "{first}.{token}",
    "{first}{yy}.{token}",
    "{first}_{token}",
]

HIGH_RISK_PATTERNS = [
    "{first}{last}admin",
    "{first}official",
    "{first}careers",
    "{first}hr",
    "support{first}{last}",
    "admin{first}",
]


def _pick(pool, weights=None):
    if weights:
        return random.choices(pool, weights=weights, k=1)[0]
    return random.choice(pool)


def _name_parts(name):
    parts = [re.sub(r"[^a-z0-9]", "", p.lower()) for p in name.split()]
    return [p for p in parts if p] or ["user"]


def _birth_year_guess(age):
    if not age:
        return random.randint(1980, 2000)
    return 2026 - int(age)


def generate_realistic_username(name, age=None, fraud=False):
    parts = _name_parts(name)

    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""

    year = str(_birth_year_guess(age))
    yy = year[-2:]

    normal_patterns = [
        f"{first}.{last}" if last else first,
        f"{first}_{last}" if last else first,
        f"{first}{year}",
        f"{first}.{last}{yy}" if last else f"{first}{yy}",
        f"{first[0]}.{last}" if last else f"{first}{random.randint(10,99)}",
        f"{first}{random.randint(10,999)}",
    ]

    normal_weights = [0.27, 0.16, 0.16, 0.18, 0.10, 0.13]

    if not fraud:
        return _pick(normal_patterns, normal_weights)

    fraud_style = random.choices(
        list(FRAUD_PERSONAS.keys()),
        weights=list(FRAUD_PERSONAS.values()),
        k=1
    )[0]

    if fraud_style == "low_signal_social":
        if random.random() < 0.82:
            return _pick(normal_patterns, normal_weights)

        token = random.choice(SUBTLE_PROFESSIONAL_TOKENS)

        pattern = random.choice(MILD_SUSPICIOUS_PATTERNS)

        return pattern.format(
            first=first,
            last=last,
            yy=yy,
            token=token,
        )

    elif fraud_style == "medium_signal_impersonator":

        token = random.choice(SUBTLE_PROFESSIONAL_TOKENS)

        options = [
            f"{first}.{last}.hr",
            f"{first}.career",
            f"{first}.{token}",
            f"{first}{yy}.{token}",
        ]

        return random.choice(options)

    else:
        pattern = random.choice(HIGH_RISK_PATTERNS)

        return pattern.format(
            first=first,
            last=last,
            yy=yy,
        )

def _institutional_probability(profession, company):
    p = (profession or "").lower()
    c = (company or "").lower()
    probability = 0.07
    if any(k in p for k in ["doctor", "surgeon", "medical"]):
        probability = 0.15
    elif any(k in p for k in ["professor", "research", "scientist"]):
        probability = 0.14
    elif any(k in p for k in ["government", "ias", "ips"]):
        probability = 0.13
    elif any(k in c for k in ["bank", "iit", "aiims", "university"]):
        probability = 0.10
    return probability


def _institutional_domain(profession, company):
    company_domains = domains_for_company(company)
    if company_domains and random.random() < 0.75:
        return _pick(company_domains)

    p = (profession or "").lower()
    if any(k in p for k in ["doctor", "surgeon", "medical"]):
        return _pick(["aiims.edu", "apollohospitals.com", "fortishealthcare.com"])
    if any(k in p for k in ["professor", "research", "scientist"]):
        return _pick(["iitd.ac.in", "iitb.ac.in", "iisc.ac.in", "du.ac.in"])
    if any(k in p for k in ["government", "ias", "ips"]):
        return _pick(["gov.in", "nic.in"])
    return _pick(EMAIL_DOMAIN_CATEGORIES["institutional"])


def generate_legitimate_email(name, age=None, profession=None, company=None, education=None):
    username = generate_realistic_username(name, age=age, fraud=False)

    if random.random() < _institutional_probability(profession, company):
        if random.random() < 0.35:
            parts = _name_parts(name)
            username = f"{parts[0][0]}.{parts[-1]}" if len(parts) > 1 else parts[0]
        domain = _institutional_domain(profession, company)
    else:
        domain = _pick(
            EMAIL_DOMAIN_CATEGORIES["personal"] + EMAIL_DOMAIN_CATEGORIES["privacy_focused"],
            [0.43, 0.20, 0.17, 0.08, 0.05, 0.035, 0.02, 0.006, 0.004, 0.003, 0.002],
        )

    return f"{username}@{domain}"


def _impersonation_email(name, profession=None, company=None):
    username_name = generate_realistic_username(name, fraud=False).split("@")[0]
    target_domains = domains_for_company(company)
    target = target_domains[0].split(".")[0] if target_domains else _pick(["aiims", "google", "infosys", "iitdelhi"])

    pattern = random.choice([
        (f"{username_name}", f"{target}-careers.com"),
        (f"doctor.{target}", "gmail.com"),
        (f"recruitment.{target}.hr", "consultantmail.com"),
        (f"{target}.admissions", f"{target}-career.org"),
    ])
    return f"{pattern[0]}@{pattern[1]}"


def generate_fraud_email(name, age=None, profession=None, company=None, education=None):
    """
    Fraud distribution intentionally includes many normal personal addresses.
    This prevents models from learning a disposable-domain shortcut.
    """
    roll = random.random()

    if roll < 0.55:
        domain = _pick(EMAIL_DOMAIN_CATEGORIES["personal"])
        username = generate_realistic_username(name, age=age, fraud=random.random() < 0.42)
    elif roll < 0.70:
        return _impersonation_email(name, profession=profession, company=company)
    elif roll < 0.97:
        domain = _pick(EMAIL_DOMAIN_CATEGORIES["typo_squatted"])
        username = generate_realistic_username(name, age=age, fraud=random.random() < 0.45)
    elif roll < 0.88:
        domain = _pick(EMAIL_DOMAIN_CATEGORIES["fake_corporate"])
        username = generate_realistic_username(name, age=age, fraud=True)
    elif roll < 0.93:
        domain = _pick(EMAIL_DOMAIN_CATEGORIES["disposable"])
        username = generate_realistic_username(name, age=age, fraud=True)
    else:
        domain = _institutional_domain(profession, company)
        username = generate_realistic_username(name, age=age, fraud=random.random() < 0.25)

    return f"{username}@{domain}"

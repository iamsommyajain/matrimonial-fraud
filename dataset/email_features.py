"""
Email credibility features for identity-consistency modeling.

The goal is soft evidence, not a blacklist. A Gmail address can be legitimate or
fraudulent; a company-looking address can be aligned, unaffiliated, typo-squatted,
or an impersonation attempt.
"""

import math
import re
from collections import Counter

from .constants import COMPANY_DOMAIN_MAP, EMAIL_DOMAIN_CATEGORIES


RISK_KEYWORDS = {
    "crypto", "forex", "fx", "investment", "loan", "rich", "single",
    "angel", "sweet", "team", "recruitment", "admission", "admissions",
    "career", "careers", "hr", "doctor", "aiims", "google", "infosys",
}


def parse_email(email):
    if not email or "@" not in email:
        return "", ""
    username, domain = email.lower().rsplit("@", 1)
    return username, domain


def get_domain_category(domain):
    for category, domains in EMAIL_DOMAIN_CATEGORIES.items():
        if domain in domains:
            return category
    return "unknown"


def username_entropy(username):
    if not username:
        return 0.0
    counts = Counter(username)
    length = len(username)
    return round(-sum((n / length) * math.log2(n / length) for n in counts.values()), 3)


def username_digit_ratio(username):
    if not username:
        return 0.0
    return sum(ch.isdigit() for ch in username) / len(username)


def has_repeated_digits(username):
    return bool(re.search(r"(\d)\1{2,}", username))


def excessive_separators(username):
    return len(re.findall(r"[._-]", username)) >= 3


def suspicious_keywords(username):
    tokens = set(re.split(r"[^a-z0-9]+", username))
    compact = re.sub(r"[^a-z]", "", username)
    return sorted(k for k in RISK_KEYWORDS if k in tokens or k in compact)


def name_alignment_score(username, profile):
    name = profile.get("name", "")
    parts = [re.sub(r"[^a-z]", "", p.lower()) for p in name.split()]
    parts = [p for p in parts if p]
    if not parts:
        return 0.3
    matches = sum(1 for p in parts if p and p in username)
    if matches >= 2:
        return 1.0
    if matches == 1:
        return 0.72
    return 0.18


def domains_for_company(company):
    if not company:
        return []
    if company in COMPANY_DOMAIN_MAP:
        return COMPANY_DOMAIN_MAP[company]
    for key, domains in COMPANY_DOMAIN_MAP.items():
        if key.lower() in company.lower() or company.lower() in key.lower():
            return domains
    return []


def domain_matches_company(domain, company):
    return domain in domains_for_company(company)


def institutional_alignment_score(domain, profession, company, education):
    if domain_matches_company(domain, company):
        return 1.0

    p = (profession or "").lower()
    c = (company or "").lower()
    e = (education or "").lower()
    category = get_domain_category(domain)

    if category != "institutional":
        return 0.45 if any(k in p for k in ["doctor", "professor", "research", "government", "ias"]) else 0.65

    if any(k in p for k in ["doctor", "medical", "surgeon"]) and any(k in domain for k in ["aiims", "hospital", "health"]):
        return 0.85
    if any(k in p for k in ["professor", "research", "scientist"]) and any(k in domain for k in ["iit", "iisc", "edu", "ac.in"]):
        return 0.85
    if any(k in p for k in ["government", "ias", "ips"]) and domain.endswith(("gov.in", "nic.in")):
        return 0.85
    if any(k in c for k in ["bank", "hdfc", "icici", "sbi"]) and any(k in domain for k in ["bank", "sbi"]):
        return 0.8
    if e and any(k in domain for k in ["edu", "ac.in"]):
        return 0.55
    return 0.35


def compute_email_consistency_score(email, profession, company, education):
    username, domain = parse_email(email)
    category = get_domain_category(domain)
    signals = []
    score = 0.78

    if category == "personal":
        score += 0.08
    elif category == "institutional":
        alignment = institutional_alignment_score(domain, profession, company, education)
        score += 0.10 * alignment
        if alignment < 0.5:
            signals.append("institutional_domain_unaligned")
            score -= 0.18
    elif category == "privacy_focused":
        signals.append("privacy_focused_domain")
        score -= 0.03
    elif category == "disposable":
        signals.append("disposable_domain")
        score -= 0.45
    elif category == "typo_squatted":
        signals.append("typo_squatted_domain")
        score -= 0.40
    elif category == "fake_corporate":
        signals.append("fake_corporate_domain")
        score -= 0.32
    else:
        signals.append("unknown_domain")
        score -= 0.08

    if domain_matches_company(domain, company):
        signals.append("domain_matches_company")
        score += 0.12

    keywords = suspicious_keywords(username)
    if keywords:
        signals.append("suspicious_username_keywords")
        score -= min(0.24, 0.05 * len(keywords))

    if username_digit_ratio(username) > 0.35 or has_repeated_digits(username):
        signals.append("excessive_digits")
        score -= 0.12

    if excessive_separators(username):
        signals.append("excessive_separators")
        score -= 0.08

    ent = username_entropy(username)
    if ent < 2.0 or ent > 4.4:
        signals.append("unusual_username_entropy")
        score -= 0.06

    return {"score": round(max(0.0, min(1.0, score)), 3), "signals": signals}


def extract_email_risk_features(email, profile):
    username, domain = parse_email(email)
    consistency = compute_email_consistency_score(
        email,
        profile.get("profession"),
        profile.get("company_name"),
        profile.get("education_level"),
    )
    keywords = suspicious_keywords(username)
    return {
        "domain_category": get_domain_category(domain),
        "domain_matches_company": domain_matches_company(domain, profile.get("company_name")),
        "institutional_alignment_score": institutional_alignment_score(
            domain,
            profile.get("profession"),
            profile.get("company_name"),
            profile.get("education_level"),
        ),
        "username_entropy": username_entropy(username),
        "suspicious_keywords": keywords,
        "typo_domain": get_domain_category(domain) == "typo_squatted",
        "disposable_domain": get_domain_category(domain) == "disposable",
        "excessive_digits": username_digit_ratio(username) > 0.35 or has_repeated_digits(username),
        "name_alignment_score": name_alignment_score(username, profile),
        "email_consistency_score": consistency["score"],
        "email_consistency_signals": consistency["signals"],
    }

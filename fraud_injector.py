"""
fraud_injector.py

Takes a legitimate profile as input and injects specific fraud signals.
Each inject_* function modifies only the fields relevant to its fraud type,
leaving everything else intact. This design lets you:

  1. Test each model (M1-M5) in isolation — each type targets one model
  2. Combine multiple injections for "multi-signal" profiles
  3. Precisely control which signals appear in injected_signals[]

FRAUD TYPES:
  Type 1 — Functional inconsistency  (M1's signal)
  Type 2 — Template bio + behavioral  (M2's signal)
  Type 3 — Image theft / deepfake     (M3's signal)
  Type 4 — Financial scam             (M4's signal, via report simulation)
  Type 5 — Coordinated ring           (M5's signal, via graph topology)
  Multi  — Combines 2-3 of the above  (tests fusion)
"""

import numpy as np
import random
import re
from datetime import datetime, timedelta
from copy import deepcopy

from constants import (
    PROFESSIONS_BY_EDUCATION, INCOME_BASE_BY_PROFESSION,
    COMPANIES_LEGITIMATE,
)
from bio_templates import BIO_TEMPLATES
from email_features import extract_email_risk_features
from email_generator import generate_fraud_email


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

# 30-word synonym map for bio paraphrasing.
# When Template bio fraud applies word substitution, these swaps make
# cosine similarity non-trivial (threshold ~0.65-0.80 not 1.0).
SYNONYM_MAP = {
    "simple":       ["humble", "ordinary", "modest", "unpretentious"],
    "caring":       ["loving", "nurturing", "affectionate", "compassionate"],
    "family":       ["household", "kin", "loved ones", "relatives"],
    "marriage":     ["matrimony", "wedlock", "partnership", "life together"],
    "values":       ["principles", "beliefs", "ethics", "morals"],
    "responsible":  ["dependable", "reliable", "accountable", "dutiful"],
    "educated":     ["qualified", "learned", "well-read", "academically sound"],
    "honest":       ["truthful", "sincere", "straightforward", "genuine"],
    "looking":      ["seeking", "searching", "hoping to find", "in search of"],
    "understand":   ["comprehend", "appreciate", "empathize", "relate to"],
    "stable":       ["secure", "settled", "established", "steady"],
    "beautiful":    ["wonderful", "lovely", "splendid", "magnificent"],
    "happy":        ["joyful", "content", "blissful", "cheerful"],
    "work":         ["job", "career", "profession", "occupation"],
    "love":         ["affection", "care", "devotion", "warmth"],
    "life":         ["existence", "journey", "path", "world"],
    "person":       ["individual", "human", "soul", "someone"],
    "good":         ["excellent", "great", "fine", "wonderful"],
    "important":    ["essential", "significant", "crucial", "vital"],
    "ready":        ["prepared", "willing", "eager", "set"],
    "partner":      ["companion", "spouse", "soulmate", "better half"],
    "believe":      ["feel", "think", "hold", "consider"],
    "traditional":  ["conventional", "orthodox", "classical", "old-fashioned"],
    "modern":       ["contemporary", "progressive", "forward-thinking", "current"],
    "strong":       ["firm", "solid", "robust", "resilient"],
    "positive":     ["optimistic", "hopeful", "constructive", "upbeat"],
    "enjoy":        ["love", "like", "relish", "take pleasure in"],
    "build":        ["create", "establish", "develop", "form"],
    "trust":        ["faith", "confidence", "reliance", "belief"],
    "respect":      ["honour", "regard", "esteem", "admire"],
}


def _substitute_words(text, rate=0.12):
    """
    Replace ~12% of substitutable words with synonyms.
    This is the core of template fraud — detectable by cosine similarity
    but not by exact string matching.
    """
    words = text.split()
    result = []
    for word in words:
        clean = word.lower().strip(".,!?;:")
        if clean in SYNONYM_MAP and random.random() < rate:
            synonym = random.choice(SYNONYM_MAP[clean])
            # Preserve capitalisation
            if word[0].isupper():
                synonym = synonym.capitalize()
            # Preserve trailing punctuation
            punct = word[len(clean):]
            result.append(synonym + punct)
        else:
            result.append(word)
    return " ".join(result)


def _make_suspicious_email(name="user"):
    """Backward-compatible fraud email helper with non-blacklist behavior."""
    return generate_fraud_email(name=name)


def _refresh_email_features(profile):
    profile["email_domain"] = profile["email"].split("@")[1]
    features = extract_email_risk_features(profile["email"], profile)
    profile["email_risk_features"] = features
    profile["email_consistency_score"] = features["email_consistency_score"]


def _burst_login_timestamps(created_at_str):
    """
    Behavioral anomaly: 0 logins for 3 weeks, then sudden burst.
    This is a classic sleeper account pattern.
    """
    created_at = datetime.fromisoformat(created_at_str)
    logins = []
    # Burst: 15-30 logins in a 48-hour window, 3-4 weeks after creation
    burst_start = created_at + timedelta(days=random.randint(21, 28))
    for _ in range(random.randint(15, 30)):
        offset_hours = random.uniform(0, 48)
        logins.append(burst_start + timedelta(hours=offset_hours))
    return [t.isoformat() for t in sorted(logins)]


def _multi_city_ips(n):
    """
    Behavioral anomaly: logins from 5+ geographically distant cities
    within a short window — impossible for a single person.
    """
    # Each prefix simulates a different city/ISP block
    prefixes = [
        f"{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
        for _ in range(random.randint(5, 8))
    ]
    ips = []
    for _ in range(n):
        prefix = random.choice(prefixes)
        ips.append(f"{prefix}.{random.randint(1, 254)}")
    return ips


def _stolen_face_embedding(pool, noise_sigma=0.02):
    """
    Return an embedding very close to one from the pool.
    Models legitimate face theft: same face, maybe slightly different photo.
    Small noise keeps it non-trivially detectable (not exact duplicate).
    """
    base = random.choice(pool)
    return (np.array(base) + np.random.normal(0, noise_sigma, len(base))).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Fraud Type 1 — Functional inconsistency  (targets M1)
# ─────────────────────────────────────────────────────────────────────────────

def inject_functional_fraud(profile):
    """
    Inject logical contradictions in structured attribute pairs.
    M1 (fuzzy membership scoring) should catch these.

    Strategy: pick 1-3 contradictions from the menu below and apply them.
    Recording exactly which signals were injected lets you do per-signal
    ablation analysis later.
    """
    p = deepcopy(profile)
    signals = []

    # Menu of possible contradictions — each is a (condition, mutation, label)
    # We randomly pick 1-3 to apply so profiles aren't all identically broken.
    possible = []

    # ── Contradiction A: age too young for claimed experience
    def too_young_for_exp():
        p["years_experience"] = p["age"] - 14  # mathematically impossible
        signals.append("age_experience_impossible")

    # ── Contradiction B: education-profession mismatch
    def edu_profession_mismatch():
        p["education_level"] = random.choice(["10th", "12th"])
        p["education_field"] = "General"
        p["college_name"] = "Local School"
        p["profession"] = random.choice(["Doctor", "Senior Software Engineer",
                                          "Research Scientist", "Professor", "Lawyer"])
        signals.append("education_profession_mismatch")

    # ── Contradiction C: income too high for company / role
    def inflated_income():
        p["annual_income_lpa"] = random.uniform(60, 200)
        p["company_name"] = f"{fake_company_name()} Solutions"  # fabricated
        signals.append("income_company_mismatch")

    # ── Contradiction D: suspicious email for high-status professional
    def professional_suspicious_email():
        p["email"] = generate_fraud_email(
            name=p.get("name", "user"),
            age=p.get("age"),
            profession=p.get("profession"),
            company=p.get("company_name"),
            education=p.get("education_level"),
        )
        _refresh_email_features(p)
        signals.append("email_identity_inconsistency")

    # ── Contradiction E: login IPs from impossible geographies
    def impossible_geo():
        # Replace all IPs with multi-country prefixes in one week
        n = len(p.get("login_timestamps", [])) or 10
        p["login_ip_list"] = _multi_city_ips(n)
        signals.append("impossible_geo_logins")

    possible = [too_young_for_exp, edu_profession_mismatch,
                inflated_income, professional_suspicious_email, impossible_geo]

    # Apply 1-3 contradictions
    n_inject = random.randint(1, 3)
    chosen = random.sample(possible, k=min(n_inject, len(possible)))
    for fn in chosen:
        fn()

    p["is_fraud"]         = True
    p["fraud_type"]       = "functional"
    p["fraud_severity"]   = "high" if len(signals) >= 3 else "medium"
    p["injected_signals"] = signals
    return p


def fake_company_name():
    """Generate a plausible-sounding but fabricated company name."""
    adjectives = ["Global", "Dynamic", "Premier", "Elite", "Advanced", "Apex", "Quantum"]
    nouns      = ["Tech", "Systems", "Solutions", "Ventures", "Innovations", "Consulting", "Services"]
    return f"{random.choice(adjectives)} {random.choice(nouns)}"


# ─────────────────────────────────────────────────────────────────────────────
# Fraud Type 2 — Template bio + behavioral anomalies  (targets M2)
# ─────────────────────────────────────────────────────────────────────────────

# Shared pool so all Type 2 profiles draw from the same templates.
# This is what makes cosine similarity detection work: many profiles will
# have bios with cosine similarity 0.70-0.90 to a template and to each other.
_TEMPLATE_POOL = [t["text"] for t in BIO_TEMPLATES]

def inject_template_bio_fraud(profile, shared_template_idx=None):
    """
    Replace bio with a word-substituted template + add behavioral anomalies.

    shared_template_idx: if provided, all fraud profiles in a ring use the
    same template (makes inter-profile similarity even higher for testing).
    """
    p = deepcopy(profile)
    signals = []

    # ── Bio substitution ──────────────────────────────────────────────────
    idx = shared_template_idx if shared_template_idx is not None else random.randint(0, len(_TEMPLATE_POOL) - 1)
    raw_template = _TEMPLATE_POOL[idx % len(_TEMPLATE_POOL)]
    # Fill slots
    raw_template = raw_template.replace("{profession}", p.get("profession", "professional"))
    raw_template = raw_template.replace("{city}", p.get("city", "the city"))
    raw_template = raw_template.replace("son/daughter", "person")
    raw_template = raw_template.replace("her/his", "their")
    raw_template = raw_template.replace("his/her", "their")
    # Apply word substitution at ~12% rate
    p["bio_text"] = _substitute_words(raw_template, rate=0.12)
    signals.append("template_bio")
    p["_template_idx"] = idx  # store for evaluation; not a model feature

    # ── Behavioral anomaly A: burst logins ────────────────────────────────
    if random.random() < 0.7:
        p["login_timestamps"] = _burst_login_timestamps(p["created_at"])
        signals.append("burst_login_pattern")

    # ── Behavioral anomaly B: all photos uploaded within 2 minutes ────────
    if random.random() < 0.6:
        base_time = datetime.fromisoformat(p["created_at"]) + timedelta(hours=1)
        p["photo_upload_dates"] = [
            (base_time + timedelta(seconds=random.randint(0, 120))).isoformat()
            for _ in range(p.get("n_photos", 3))
        ]
        signals.append("rapid_photo_upload")

    # ── Behavioral anomaly C: excessive edits ────────────────────────────
    if random.random() < 0.5:
        created_at = datetime.fromisoformat(p["created_at"])
        p["profile_edit_count"] = random.randint(15, 40)
        p["edit_timestamps"] = [
            (created_at + timedelta(
                days=random.randint(0, 2), hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )).isoformat()
            for _ in range(p["profile_edit_count"])
        ]
        signals.append("excessive_edits")

    p["is_fraud"]         = True
    p["fraud_type"]       = "template_bio"
    p["fraud_severity"]   = "medium"
    p["injected_signals"] = signals
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fraud Type 3 — Image theft / visual fraud  (targets M3)
# ─────────────────────────────────────────────────────────────────────────────

# Global pool of "stolen" embeddings — shared across Type 3 profiles.
# When multiple profiles share near-identical embeddings, M3's FAISS
# similarity search will flag them.
STOLEN_EMBEDDING_POOL = [
    np.random.normal(0, 1, 128).tolist() for _ in range(50)
]


def inject_image_fraud(profile):
    """
    Inject visual fraud signals:
      A. Stolen photos: embeddings near-identical to other profiles
      B. Multiple identities: embeddings in one profile from different distributions
      C. Deepfake flag: set authenticity_score low
      D. Suspicious EXIF: stripped metadata, GIMP software
    """
    p = deepcopy(profile)
    signals = []
    n_photos = p.get("n_photos", 3)

    fraud_subtype = random.choice(["stolen", "multiple_identity", "deepfake", "mixed"])

    if fraud_subtype == "stolen":
        # All photos come from the same stolen source
        stolen_base = random.choice(STOLEN_EMBEDDING_POOL)
        p["face_embeddings"] = [
            _stolen_face_embedding([stolen_base]) for _ in range(n_photos)
        ]
        signals.append("stolen_images")

    elif fraud_subtype == "multiple_identity":
        # Photos in this profile come from MULTIPLE different real people.
        # Realistic scenario: catfish assembling photos from different victims.
        # Each "half" of photos has a different base embedding.
        base_a = np.random.normal(0, 1, 128).tolist()
        base_b = np.random.normal(0, 1, 128).tolist()
        split = n_photos // 2
        p["face_embeddings"] = (
            [_stolen_face_embedding([base_a]) for _ in range(split)] +
            [_stolen_face_embedding([base_b]) for _ in range(n_photos - split)]
        )
        signals.append("multiple_identity_in_profile")

    elif fraud_subtype == "deepfake":
        # Mark as AI-generated — in real system this comes from FaceForensics++
        p["face_embeddings"] = [np.random.normal(0, 1, 128).tolist() for _ in range(n_photos)]
        p["deepfake_score"] = [random.uniform(0.75, 0.99) for _ in range(n_photos)]
        signals.append("deepfake_detected")

    else:  # mixed
        stolen_base = random.choice(STOLEN_EMBEDDING_POOL)
        p["face_embeddings"] = [_stolen_face_embedding([stolen_base]) for _ in range(n_photos)]
        p["deepfake_score"] = [random.uniform(0.5, 0.85) for _ in range(n_photos)]
        signals.append("stolen_images")
        signals.append("partial_deepfake")

    # EXIF anomalies — almost always present in image fraud
    p["exif_data"] = [
        {
            "device": None,
            "software": random.choice(["GIMP 2.10", "Adobe Photoshop 2023", "Unknown", None]),
            "gps_stripped": True,
            "timestamp_consistent": False,
            "source_suspicious": True,
        }
        for _ in range(n_photos)
    ]
    signals.append("suspicious_exif")

    p["is_fraud"]         = True
    p["fraud_type"]       = "image_theft"
    p["fraud_severity"]   = "high"
    p["injected_signals"] = signals
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fraud Type 4 — Financial scam  (targets M4 via simulated reports)
# ─────────────────────────────────────────────────────────────────────────────

REPORT_CATEGORIES = [
    "financial_solicitation",
    "identity_mismatch",
    "scripted_conversation",
    "off_platform_pressure",
    "fake_emergency_request",
    "gift_card_request",
]


def inject_financial_scam_signals(profile):
    """
    Financial scam profiles look clean at onboarding — M1/M2/M3 won't catch
    them. The signal lives in M4's crowdsourced report timeline.

    We simulate:
      - 3-7 reports from different users within 14 days of first contact
      - Reports categorized as financial/pressure
      - Sudden inactivity after reports surge
      - Suspicious pattern: profile never accepts matches, only initiates
    """
    p = deepcopy(profile)
    signals = []

    created_at = datetime.fromisoformat(p["created_at"])

    # ── Interaction pattern: mass outreach, zero reciprocation ────────────
    p["messages_sent"]     = random.randint(50, 200)   # far more than legitimate
    p["messages_received"] = random.randint(0, 5)      # victims don't respond much
    p["match_requests_sent"] = random.randint(30, 80)
    p["match_accepts"]     = 0                         # never accepts incoming
    p["unique_contacts"]   = random.randint(40, 100)
    signals.append("high_outreach_low_response")

    # ── Simulated reports (stored as metadata — M4 reads this) ───────────
    n_reports = random.randint(3, 7)
    first_contact_day = random.randint(3, 10)  # days after creation
    reports = []
    for _ in range(n_reports):
        days_after_contact = random.randint(1, 14)
        report_ts = created_at + timedelta(
            days=first_contact_day + days_after_contact,
            hours=random.randint(0, 23)
        )
        reports.append({
            "report_timestamp": report_ts.isoformat(),
            "category":         random.choice(["financial_solicitation",
                                               "off_platform_pressure",
                                               "fake_emergency_request"]),
            "reporter_verified": random.random() < 0.7,  # 70% from verified users
            "reporter_account_age_days": random.randint(30, 500),
        })
    p["simulated_reports"] = reports
    signals.append(f"reports_count_{n_reports}")
    signals.append("financial_report_category")

    # ── Sudden inactivity after reports ──────────────────────────────────
    last_active = created_at + timedelta(days=first_contact_day + 15)
    p["last_active_at"] = last_active.isoformat()
    signals.append("sudden_inactivity_post_reports")

    p["is_fraud"]         = True
    p["fraud_type"]       = "financial_scam"
    p["fraud_severity"]   = "high"
    p["injected_signals"] = signals
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fraud Type 5 — Coordinated ring  (targets M5 via graph topology)
# ─────────────────────────────────────────────────────────────────────────────
# Note: full graph construction is in graph_generator.py.
# This function marks the profile as a ring member and sets its interaction
# statistics. The actual edges are built when the full graph is assembled.

def inject_ring_signals(profile, ring_id, role="satellite"):
    """
    Mark a profile as part of a coordinated fraud ring.

    role:
      "hub"      — the central node that contacts hundreds of victims
      "satellite" — peripheral ring members that relay or reinforce the hub
      "relay"    — intermediate nodes in a relay chain

    Ring profiles are characterized by:
      - Very similar creation timestamps (created in a batch)
      - Star or relay topology in the interaction graph
      - No genuine two-way conversations
      - Often idle except during coordinated campaigns
    """
    p = deepcopy(profile)
    signals = []

    p["ring_id"] = ring_id
    p["ring_role"] = role

    if role == "hub":
        # Hub sends huge volume to legitimate victims
        p["messages_sent"]      = random.randint(200, 500)
        p["messages_received"]  = random.randint(5, 20)
        p["match_requests_sent"]= random.randint(100, 300)
        p["match_accepts"]      = 0
        p["unique_contacts"]    = random.randint(150, 400)
        signals.append("star_hub_topology")
        signals.append("mass_outreach")

    elif role == "relay":
        # Relay passes messages between hub and victims
        p["messages_sent"]      = random.randint(30, 80)
        p["messages_received"]  = random.randint(30, 80)  # balanced (relaying)
        p["unique_contacts"]    = random.randint(10, 30)
        signals.append("relay_chain_node")

    else:  # satellite
        # Satellites provide social proof or backup scam accounts
        p["messages_sent"]      = random.randint(10, 50)
        p["messages_received"]  = random.randint(0, 5)
        p["unique_contacts"]    = random.randint(5, 20)
        signals.append("ring_satellite")

    signals.append(f"ring_{ring_id}")

    p["is_fraud"]         = True
    p["fraud_type"]       = "coordinated_ring"
    p["fraud_severity"]   = "high" if role == "hub" else "medium"
    p["injected_signals"] = signals
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Multi-signal fraud  (combines 2-3 types for the hardest cases)
# ─────────────────────────────────────────────────────────────────────────────

def inject_multi_signal_fraud(profile, shared_template_idx=None):
    """
    Apply 2-3 fraud injection types to a single profile.
    These are the hardest cases for any single model to catch —
    they demonstrate why score fusion (Stage 3) is necessary.
    """
    p = deepcopy(profile)

    # Pick 2-3 combinations
    combo = random.choice([
        ["functional", "template_bio"],
        ["functional", "image"],
        ["template_bio", "image"],
        ["functional", "template_bio", "image"],
        ["functional", "financial"],
        ["template_bio", "financial"],
    ])

    all_signals = []

    if "functional" in combo:
        p = inject_functional_fraud(p)
        all_signals.extend(p["injected_signals"])

    if "template_bio" in combo:
        p = inject_template_bio_fraud(p, shared_template_idx)
        all_signals.extend(p["injected_signals"])

    if "image" in combo:
        p = inject_image_fraud(p)
        all_signals.extend(p["injected_signals"])

    if "financial" in combo:
        p = inject_financial_scam_signals(p)
        all_signals.extend(p["injected_signals"])

    p["is_fraud"]         = True
    p["fraud_type"]       = "multi"
    p["fraud_severity"]   = "high"
    p["injected_signals"] = list(set(all_signals))  # deduplicate
    return p

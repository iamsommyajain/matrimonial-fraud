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
import re
from datetime import datetime, timedelta
from faker import Faker
from uuid import uuid4

from constants import (
    RELIGIONS, RELIGION_WEIGHTS, CASTES_BY_RELIGION,
    MOTHER_TONGUES, TONGUE_WEIGHTS, GENDERS, GENDER_WEIGHTS,
    EDUCATION_LEVELS, EDUCATION_WEIGHTS, EDUCATION_FIELDS, COLLEGES,
    PROFESSIONS_BY_EDUCATION, COMPANIES_LEGITIMATE, INCOME_BASE_BY_PROFESSION,
    LEGITIMATE_EMAIL_DOMAINS, ALL_CITIES, CITY_WEIGHTS, INDIAN_STATES,
    COUNTRIES, COUNTRY_WEIGHTS, HOBBIES_POOL,
)
from bio_templates import BIO_TEMPLATES, PARTNER_PREF_TEMPLATES

fake = Faker('en_IN')
rng = np.random.default_rng()   # single shared RNG — seed at call site


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick(pool, weights=None):
    """Weighted random pick from a list."""
    if weights:
        return random.choices(pool, weights=weights, k=1)[0]
    return random.choice(pool)


def _realistic_experience(age, education_level):
    """
    Compute a realistic years_of_experience given age and education.

    Logic:
      - Minimum working age after education completion:
          10th  → 16 (school dropout + vocational)
          12th  → 18
          Graduate → 21-22
          Post Graduate → 23-24
          PhD → 27-28
      - Add small Gaussian noise so it doesn't look generated.
      - Clamp to [0, age - min_work_age].
    """
    min_work_age = {
        "10th": 16, "12th": 18, "Graduate": 22,
        "Post Graduate": 24, "PhD": 28,
    }[education_level]

    max_possible = max(0, age - min_work_age)
    # Most people spend 0-2 years finding first job after education
    expected = max(0, max_possible - random.randint(0, 2))
    noise = int(np.random.normal(0, 1))
    return max(0, min(max_possible, expected + noise))


def _realistic_income(profession, years_experience):
    """
    Compute annual income in LPA (Lakhs Per Annum) from profession and exp.
    Uses Indian market rates roughly calibrated to 2024 values.
    """
    params = INCOME_BASE_BY_PROFESSION.get(profession, INCOME_BASE_BY_PROFESSION["DEFAULT"])
    income = (
        params["base"]
        + years_experience * params["per_year_exp"]
        + np.random.normal(0, params["sigma"])
    )
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


def _pick_email(profession, name):
    """
    Legitimate professionals sometimes use company/institution emails.
    Doctors, professors, and govt employees more often have institutional mail.
    """
    p = profession.lower()
    has_institutional = (
        any(k in p for k in ["professor", "doctor", "government", "ias"]) and
        random.random() < 0.35  # 35% chance of institutional email
    )
    if has_institutional:
        slug = re.sub(r'[^a-z]', '', name.lower())[:10]
        institution_domains = ["aiims.edu", "gov.in", "iit.ac.in", "nit.ac.in",
                               "hospital.org", "university.ac.in"]
        return f"{slug}@{_pick(institution_domains)}"
    else:
        slug = re.sub(r'[^a-z0-9]', '', name.lower().replace(' ', '.'))[:12]
        suffix = str(random.randint(1, 999)) if random.random() < 0.4 else ""
        return f"{slug}{suffix}@{_pick(LEGITIMATE_EMAIL_DOMAINS)}"


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

def generate_legitimate_profile(created_at=None):
    """
    Generate one fully coherent legitimate matrimonial profile.

    Returns a flat dict with all fields needed for M1–M5 training.
    """
    if created_at is None:
        # Profiles created anywhere in the last 6 months
        created_at = datetime.now() - timedelta(days=random.randint(1, 180))

    # ── Step 1: Core demographics ──────────────────────────────────────────
    gender       = _pick(GENDERS, GENDER_WEIGHTS)
    religion     = _pick(RELIGIONS, RELIGION_WEIGHTS)
    caste        = _pick(CASTES_BY_RELIGION[religion])
    mother_tongue= _pick(MOTHER_TONGUES, TONGUE_WEIGHTS)
    age          = random.randint(22, 50)
    height_cm    = (
        random.randint(165, 188) if gender == "Male"
        else random.randint(152, 172)
    )

    # Generate a name matching gender and region
    if gender == "Male":
        name = fake.name_male()
    else:
        name = fake.name_female()

    city    = _pick(ALL_CITIES, CITY_WEIGHTS)
    state   = _pick(INDIAN_STATES)
    country = _pick(COUNTRIES, COUNTRY_WEIGHTS)

    # ── Step 2: Education (determines downstream fields) ───────────────────
    education_level = _pick(EDUCATION_LEVELS, EDUCATION_WEIGHTS)
    education_field = _pick(EDUCATION_FIELDS[education_level])
    college         = _pick(COLLEGES[education_level])

    # ── Step 3: Profession (conditional on education) ─────────────────────
    profession      = _pick(PROFESSIONS_BY_EDUCATION[education_level])
    company         = _pick_company(profession)

    # ── Step 4: Experience and income (conditional on age + profession) ───
    years_experience = _realistic_experience(age, education_level)
    annual_income_lpa = _realistic_income(profession, years_experience)

    # ── Step 5: Contact details (conditional on profession) ───────────────
    email = _pick_email(profession, name)

    # ── Step 6: Bio and preferences ───────────────────────────────────────
    bio_text          = _generate_bio(name, profession, city, gender)
    partner_prefs     = _generate_partner_prefs(age, city)
    about_family      = (
        f"We are a {_pick(['close-knit', 'traditional', 'modern yet rooted', 'simple'])} "
        f"family based in {city}. My father is {_pick(['retired', 'a businessman', 'a government employee', 'a professional'])} "
        f"and my mother is {_pick(['a homemaker', 'a teacher', 'also working'])}. "
        f"We have {random.randint(1, 3)} sibling(s) and share a very warm household."
    )
    hobbies = random.sample(HOBBIES_POOL, k=random.randint(3, 6))

    # ── Step 7: Behavioral / temporal signals ─────────────────────────────
    timestamps = _generate_timestamps(created_at)

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
    # These are per-profile stats; the full graph is assembled in graph_generator.py
    messages_sent     = random.randint(0, 30)
    messages_received = int(messages_sent * random.uniform(0.3, 1.5))
    match_requests    = random.randint(0, 15)
    match_accepts     = int(match_requests * random.uniform(0.1, 0.5))
    unique_contacts   = random.randint(1, min(20, messages_sent + 1))

    # ── Assemble ─────────────────────────────────────────────────────────
    return {
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
        "country":              country,

        # Education
        "education_level":      education_level,
        "education_field":      education_field,
        "college_name":         college,

        # Profession
        "profession":           profession,
        "company_name":         company,
        "years_experience":     years_experience,
        "annual_income_lpa":    annual_income_lpa,
        "email":                email,
        "email_domain":         email.split("@")[1],

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
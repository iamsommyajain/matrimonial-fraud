"""
feature_extractor.py

Pulls and normalises the M1-relevant fields from a raw profile dict.
This is a thin layer — no scoring logic here, just clean extraction
with safe defaults for missing fields.

Returns a typed dict that scorer.py and fuzzy_rules.py can consume
without doing any type coercion themselves.
"""

from typing import Any


# Fields we need — with their types and safe defaults
_FIELD_DEFAULTS = {
    "age":                  (int,   None),
    "education_level":      (str,   None),
    "years_experience":     (int,   None),
    "profession":           (str,   None),
    "annual_income_lpa":    (float, None),
    "email_domain":         (str,   None),
    "email":                (str,   None),
}


def extract_m1_features(profile: dict) -> dict:
    """
    Extract and type-normalise M1 features from a raw profile dict.

    Handles:
      - Missing keys         → None (scorer treats as unknown → 0 penalty)
      - Wrong types          → silent cast with fallback to None
      - email without domain → derives domain from email field
      - Negative values      → clamped to 0

    Returns:
        {
          "age":               int | None,
          "education_level":   str | None,   e.g. "Graduate"
          "years_experience":  int | None,
          "profession":        str | None,
          "annual_income_lpa": float | None,
          "email_domain":      str | None,   e.g. "gmail.com"
        }
    """
    features = {}

    for field, (cast_type, default) in _FIELD_DEFAULTS.items():
        raw = profile.get(field, default)
        features[field] = _safe_cast(raw, cast_type)

    # Derive email_domain from email if not directly present
    if features.get("email_domain") is None and features.get("email"):
        features["email_domain"] = _extract_domain(features["email"])

    # Clamp numeric fields to non-negative
    for field in ("age", "years_experience", "annual_income_lpa"):
        val = features.get(field)
        if val is not None and val < 0:
            features[field] = 0

    # Normalise education level string
    if features.get("education_level"):
        features["education_level"] = _normalise_education(features["education_level"])

    # Normalise email domain to lowercase
    if features.get("email_domain"):
        features["email_domain"] = features["email_domain"].lower().strip()

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_cast(value: Any, cast_type: type) -> Any:
    """Cast value to cast_type; return None on failure."""
    if value is None:
        return None
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return None


def _extract_domain(email: str) -> str | None:
    """Pull domain from email string."""
    try:
        return email.strip().split("@")[1].lower()
    except (IndexError, AttributeError):
        return None


# Aliases people might use → canonical form
_EDU_ALIASES = {
    "10":               "10th",
    "10th":             "10th",
    "matriculation":    "10th",
    "sslc":             "10th",
    "12":               "12th",
    "12th":             "12th",
    "intermediate":     "12th",
    "hsc":              "12th",
    "puc":              "12th",
    "graduate":         "Graduate",
    "graduation":       "Graduate",
    "ug":               "Graduate",
    "bachelor":         "Graduate",
    "bachelors":        "Graduate",
    "b.tech":           "Graduate",
    "b.e":              "Graduate",
    "b.sc":             "Graduate",
    "b.com":            "Graduate",
    "b.a":              "Graduate",
    "mbbs":             "Post Graduate",   # MBBS is 5.5 years — treated as PG
    "post graduate":    "Post Graduate",
    "postgraduate":     "Post Graduate",
    "pg":               "Post Graduate",
    "masters":          "Post Graduate",
    "master":           "Post Graduate",
    "mba":              "Post Graduate",
    "m.tech":           "Post Graduate",
    "m.sc":             "Post Graduate",
    "m.com":            "Post Graduate",
    "m.a":              "Post Graduate",
    "mca":              "Post Graduate",
    "phd":              "PhD",
    "ph.d":             "PhD",
    "doctorate":        "PhD",
    "doctoral":         "PhD",
}


def _normalise_education(raw: str) -> str:
    """Map raw education string → canonical level."""
    normalised = raw.strip().lower()
    return _EDU_ALIASES.get(normalised, raw)   # unknown → pass through unchanged
